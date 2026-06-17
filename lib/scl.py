import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import numpy as np

def has_inf_or_nan(x):
    return torch.isinf(x).max().item(), torch.isnan(x).max().item()

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_all_classes = 2
        self.num_real_classes = self.num_all_classes
        self.cross_temperature = 0.1
        self.base_temperature = 1.0
        self.label_scaling_mode = 'nn'
        self.cross_scale_contrast = False
        self.dominant_mode = 'all'
        self.eps = torch.tensor(1e-10)
        self.metadata = {}
        self.min_views_per_class = 10
        self.max_views_per_class = 300
        self.max_features_total = 10000
        self.log_this_step = False
        self._scale = None

        if self.label_scaling_mode == 'nn':
            assert(self.dominant_mode == 'all'), \
                'cannot use label_scaling_mode: "{}" with dominant_mode: "{}" -' \
                ' only "all" is allowed'.format(self.label_scaling_mode, self.dominant_mode)

    def forward(self, label_x: torch.Tensor, label_lf: torch.Tensor, label_hf: torch.Tensor, features: torch.Tensor):
        flag_error_x_lf = flag_error_x_hf = flag_error_lf = flag_error_hf = False
        
        dominant_classes_x = label_x
        dominant_classes_lf = label_lf
        dominant_classes_hf = label_hf
        sampled_features_x_lf, sampled_labels_x_lf, flag_error_x_lf = self.sample_anchors(dominant_classes_lf, features[0])
        sampled_features_x_hf, sampled_labels_x_hf, flag_error_x_hf = self.sample_anchors(dominant_classes_hf, features[0])
        sampled_features_lf, sampled_labels_lf, flag_error_lf = self.sample_anchors(dominant_classes_lf, features[1])
        sampled_features_hf, sampled_labels_hf, flag_error_hf = self.sample_anchors(dominant_classes_hf, features[2])

        if flag_error_x_lf and flag_error_lf and flag_error_hf and flag_error_x_hf: 
            loss = features-features
            loss = loss.mean()
        else:
            loss_lf = self.calculate_loss(sampled_features_x_lf, sampled_labels_x_lf, sampled_features_lf, sampled_labels_lf)
            loss_hf = self.calculate_loss(sampled_features_x_hf, sampled_labels_x_hf, sampled_features_hf, sampled_labels_hf)
            loss_x = self.calculate_loss(sampled_features_x_lf, sampled_labels_x_lf, sampled_features_x_lf, sampled_labels_x_lf)
            loss = loss_x + loss_lf + loss_hf
        return loss

    def calculate_loss(self, feats1, labels1, feats2, labels2):
        loss = []
        for feat1, label1, feat2, label2 in zip(feats1, labels1, feats2, labels2):
                loss.append(self.contrastive_loss(feat1, label1, feat2, label2))

        return sum(loss) / (len(loss)+1e-10)
        
    def _select_views_per_class(self, min_views, total_cls, cls_in_batch, cls_counts_in_batch):
        if self.max_views_per_class == 1:
            views_per_class = min_views
        else:
            views_per_class = max(min_views, self.min_views_per_class)
            views_per_class = min(views_per_class, self.max_views_per_class)
            if views_per_class == self.max_views_per_class:
                self.log_this_step = True
        if views_per_class * total_cls > self.max_features_total:
            views_per_class = self.max_features_total // total_cls
            self.log_this_step = True
        return views_per_class

    def sample_anchors_fast(self, dominant_classes, features):

        flag_error = False
        n = dominant_classes.shape[0]  # batch size
        c = features.shape[1]  # feature space dimensionality
        features = features.view(n, c, -1)
        dominant_classes = dominant_classes.view(n, -1)  # flatten   # flatten n,1,h,w --> n,h*w
        skip_ids = []
        cls_in_batch = []  # list of lists each containing classes in an image of the batch
        cls_counts_in_batch = []  # list of lists each containing classes in an image of the batch

        classes_ids = torch.arange(start=0, end=self.num_all_classes, step=1, device=dominant_classes.device)
        compare = dominant_classes.unsqueeze(-1) == classes_ids.unsqueeze(0).unsqueeze(0)# n, hw, 1 == 1, 1, n_c => n,hw,n_c
        cls_counts = compare.sum(1) # n, n_c

        present_inds = torch.where(cls_counts[:, :-1] >= self.min_views_per_class) # ([0,...,n-1], [prese   nt class ids])
        batch_inds, cls_in_batch = present_inds

        min_views = torch.min(cls_counts[present_inds])
        total_cls = cls_in_batch.shape[0]

        views_per_class = self._select_views_per_class(min_views, total_cls, cls_in_batch, cls_counts_in_batch)
        sampled_features = torch.zeros((total_cls, c, views_per_class), dtype=torch.float).cuda()
        sampled_labels = torch.zeros(total_cls, dtype=torch.float).cuda()

        for i in range(total_cls):
            indices_from_cl_fast = compare[batch_inds[i], :, cls_in_batch[i]].nonzero().squeeze()
            random_permutation = torch.randperm(indices_from_cl_fast.shape[0]).cuda()
            sampled_indices_from_cl = indices_from_cl_fast[random_permutation[:views_per_class]]
            sampled_features[i] = features[batch_inds[i], :, sampled_indices_from_cl]
            sampled_labels[i] = cls_in_batch[i]

        return sampled_features, sampled_labels, flag_error

    def sample_anchors(self, dominant_classes, features):

        flag_error = False
        n = dominant_classes.shape[0]  # batch size
        c = features.shape[1]  # feature space dimensionality
        features = features.view(n, c, -1)
        dominant_classes = dominant_classes.view(n, -1)
        skip_num = 0
        
        sampled_features = []
        sampled_labels = []
        for i in range(n):
            sampled_feature = torch.empty(0, c, dtype=torch.float).cuda()
            sampled_label = torch.empty(0, dtype=torch.float).cuda()
            y_i = dominant_classes[i].squeeze()
            cls_in_y_i, cls_counts_in_y_i = torch.unique(y_i, return_counts=True)
            
            min_views_current = torch.min(cls_counts_in_y_i).item()
            if min_views_current > self.min_views_per_class:
                min_views_current = min(min_views_current, self.max_views_per_class)
                
                for cl in cls_in_y_i:
                    indices_from_cl = (y_i == cl).nonzero().squeeze()
                    random_permutation = torch.randperm(indices_from_cl.shape[0])
                    sampled_indices_from_cl = indices_from_cl[random_permutation[:min_views_current]]
                    feat = features[i, :, sampled_indices_from_cl]
                    feat = torch.transpose(feat, 0, 1)
                    label = cl.repeat(min_views_current)
                    
                    sampled_feature = torch.cat((sampled_feature, feat), dim=0)
                    sampled_label = torch.cat((sampled_label, label), dim=0)
                    
                sampled_features.append(sampled_feature)
                sampled_labels.append(sampled_label)
        
                    
            else:
                skip_num+=1
        if skip_num == n:
            flag_error = True
                
        return sampled_features, sampled_labels, flag_error
 
    def contrastive_loss(self, feats1, labels1, feats2, labels2):

        # prepare feats
        feats_flat1 = torch.nn.functional.normalize(feats1, p=2, dim=1)  # L2 normalization
        # feats_flat1 = feats1  # L2 normalization
        labels1 = labels1.unsqueeze(1)

        feats_flat2 = torch.nn.functional.normalize(feats2, p=2, dim=1)  # L2 normalization
        # feats_flat2 = feats2
        labels2 = labels2.unsqueeze(1)


        pos_mask, neg_mask = self.get_masks(labels1, labels2)
        dot_product = torch.div(torch.matmul(feats_flat1, torch.transpose(feats_flat2, 0, 1)), self.cross_temperature)
        loss2 = self.InfoNce_loss(pos_mask, neg_mask, dot_product)
        return loss2

    @staticmethod
    def get_masks(lbl1, lbl2):

        # extract mask indicating same class samples
        pos_mask = torch.eq(lbl1, torch.transpose(lbl2, 0, 1)).float()  # mask T-T  # indicator of positives
        pos_sums = pos_mask.sum(1)

        neg_mask = (1 - pos_mask)  # indicator of negatives
        return pos_mask, neg_mask

    def InfoNce_loss(self, pos, neg, dot):

        logits = dot 

        neg_logits = torch.exp(logits) * neg
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)
       
        pos_sums = pos.sum(1)
        ones = torch.ones(size=pos_sums.size())
        norm = torch.where(pos_sums > 0, pos_sums, ones.to(pos.device))
        mean_log_prob_pos = (pos * log_prob).sum(1) / norm   
        
        loss = - mean_log_prob_pos

        loss = loss.mean()

        if has_inf_or_nan(loss)[0] or has_inf_or_nan(loss)[1]:
            print('\n inf found in loss with positives {} and Negatives {}'.format(pos.sum(1), neg.sum(1)))
        return loss
