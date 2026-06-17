import os
import logging
import numpy as np
from datetime import datetime
from lib.fsanet import FSANet
from lib.scl import ContrastiveLoss
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim
from torch.autograd import Variable
from torchvision.utils import make_grid
import eval.python.metrics as Measure
from utils.utils import *
from utils.dataloader import get_loader, valid_dataset



def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def train(opt):
        
    save_path = os.path.join(opt.Valid.Checkpoint.checkpoint_dir, opt.Model.name)
    os.makedirs(save_path, exist_ok=True)
        
    # logging
    logger = logging.getLogger(opt.Model.name)
    file_handler = logging.FileHandler(os.path.join(save_path, "log.log"))
    formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    logger.info(">>> current mode: network-train/val")
    logger.info('>>> config: {}'.format(opt))

    # ---- build models ----
    torch.cuda.set_device(opt.Train.DeviceNum)  # set your gpu device 
    model = eval(opt.Model.net)().cuda()
    if opt.Train.DataParallel:
        model = torch.nn.DataParallel(model)
        
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.Train.Optimizer.lr)
    
    cosine_schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=opt.Train.Scheduler.T_0, T_mult=opt.Train.Scheduler.T_mult, eta_min=opt.Train.Scheduler.eta_min)
    
    train_path = os.path.join(opt.Train.Dataset.root, 'TrainDataset')
    train_loader = get_loader(train_path, batch_size=opt.Train.Dataloader.batch_size, image_size=opt.Train.Dataloader.image_size,
                            shuffle=opt.Train.Dataloader.shuffle, num_workers=opt.Train.Dataloader.num_workers, 
                            pin_memory=opt.Train.Dataloader.pin_memory, augmentation=opt.Train.Dataloader.augmentation)
    total_step = len(train_loader)
    
    logger.info("Start Training")
    print("Start Training")

    for epoch in range(1, opt.Train.Scheduler.epoch+1):
        model.train()
        scl_loss = ContrastiveLoss()
        for i, pack in enumerate(train_loader):
            optimizer.zero_grad()
            
            # ---- data prepare ----
            images, gts, edges, gts_b, edges_b = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            edges = Variable(edges).cuda()
            gts_b = Variable(gts_b).cuda()
            edges_b = Variable(edges_b).cuda()
            
            gt1 = F.interpolate(gts, size=(opt.Train.Dataloader.image_size//4, opt.Train.Dataloader.image_size//4), mode='bilinear', align_corners=True)
            eg1 = F.interpolate(edges, size=(opt.Train.Dataloader.image_size//4, opt.Train.Dataloader.image_size//4), mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gts, size=(opt.Train.Dataloader.image_size//8, opt.Train.Dataloader.image_size//8), mode='bilinear', align_corners=True)
            eg2 = F.interpolate(edges, size=(opt.Train.Dataloader.image_size//8, opt.Train.Dataloader.image_size//8), mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gts, size=(opt.Train.Dataloader.image_size//16, opt.Train.Dataloader.image_size//16), mode='bilinear', align_corners=True)
            eg3 = F.interpolate(edges, size=(opt.Train.Dataloader.image_size//16, opt.Train.Dataloader.image_size//16), mode='bilinear', align_corners=True)
            
            gt_b = F.interpolate(gts_b, size=(opt.Train.Dataloader.image_size//4, opt.Train.Dataloader.image_size//4), mode='bilinear', align_corners=True)
            eg_b = F.interpolate(edges_b, size=(opt.Train.Dataloader.image_size//4, opt.Train.Dataloader.image_size//4), mode='bilinear', align_corners=True)
            # ---- forward ----
            P, LF, HF, Proj = model(images)
            # ---- loss function ----
            loss_seg = structure_loss(P, gts)
            loss_lf_1 = dice_loss(LF[0], gt1)
            loss_lf_2 = dice_loss(LF[1], gt2)
            loss_lf_3 = dice_loss(LF[2], gt3)
            loss_lf = 0.5 * loss_lf_1 + 0.3 * loss_lf_2 + 0.2 * loss_lf_3
            loss_hf_1 = dice_loss(HF[0], eg1)
            loss_hf_2 = dice_loss(HF[1], eg2)
            loss_hf_3 = dice_loss(HF[2], eg3)
            loss_hf = 0.5 * loss_hf_1 + 0.3 * loss_hf_2 + 0.2 * loss_hf_3
            loss_scl = scl_loss(gt_b.long(), gt_b.long(), eg_b.long(), Proj)
            
            loss = loss_seg + loss_lf + 2 * loss_hf + 0.1 * loss_scl
            # ---- backward ----
            loss.backward()
            
            clip_gradient(optimizer, opt.Train.Optimizer.clip)
            optimizer.step()
            
            # ---- train visualization ----
            if i % 20 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:.8f}, Total_loss: {:.4f} loss_pred: {:.4f} loss_sem: {:.4f} loss_edge: {:0.4f}  loss_contrast_x: {:.4f}'.
                    format(datetime.now(), epoch, opt.Train.Scheduler.epoch, i, total_step, optimizer.param_groups[0]['lr'], loss.data, loss_pred.data, loss_sem.data, loss_edge.data, loss_contrast_x.data))
                logger.info('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:.8f}, Total_loss: {:.4f} loss_pred: {:.4f} loss_sem: {:.4f} loss_edge: {:0.4f}  loss_contrast_x: {:.4f}'.
                    format(epoch, opt.Train.Scheduler.epoch, i, total_step, optimizer.param_groups[0]['lr'], loss.data, loss_pred.data, loss_sem.data, loss_edge.data, loss_contrast_x.data))
        cosine_schedule.step()    
        if epoch >= opt.Valid.Checkpoint.checkpoint_epoch:
            valid(opt, logger, epoch, save_path, model)
                
                
def valid(opt, logger, epoch, save_path, model):
    """
    validation function
    """
    global best_score, best_epoch
    dice = {key: 0 for key in opt.Valid.Dataset.names}
    iou = {key: 0 for key in opt.Valid.Dataset.names}
    print('{} Epoch [{:03d}/{:03d}], Start Validating...'.format(datetime.now(), epoch, opt.Train.Scheduler.epoch))
    logger.info('Epoch [{:03d}/{:03d}],  Start Validating...'.format(epoch, opt.Train.Scheduler.epoch))

    for model_name in opt.Valid.Dataset.names:
        valid_path = os.path.join(opt.Valid.Dataset.root, 'TestDataset', model_name)
        dataset = valid_dataset(valid_path, opt.Valid.Dataloader.image_size)
        valid_loader = data.data_loader = data.DataLoader(dataset=dataset, 
                                                            batch_size=opt.Valid.Dataloader.batch_size, 
                                                            shuffle=opt.Valid.Dataloader.shuffle)
        evaluator = Evaluator()

        model.eval()  
        with torch.no_grad():  
            for pack in valid_loader:
                image, gt, _ = pack
                image = image.cuda()
                gt = gt.cuda()
                
                P = model(image)
                res = P[0].sigmoid()
                evaluator.update_batch(res, gt)

        dice[model_name], iou[model_name] = evaluator.show()
        print("Model Name: {}, Dice: {}, IoU: {}".format(model_name, dice[model_name], iou[model_name]))
        logger.info("Model Name: {}, Dice: {}, IoU: {}".format(model_name, dice[model_name], iou[model_name]))
        
    print('{} Epoch [{:03d}/{:03d}], End Validating.'.format(datetime.now(), epoch, opt.Train.Scheduler.epoch))
    logger.info('Epoch [{:03d}/{:03d}],  End Validating.'.format(epoch, opt.Train.Scheduler.epoch))
    current_score = sum(dice.values()) + sum(iou.values())
    if epoch <= opt.Valid.Checkpoint.checkpoint_epoch:
        best_score = current_score
        best_epoch = epoch
    else:
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + '/' + '{}_best.pth'.format(opt.Model.name))
            logger.info('Saving Snapshot: {}_best_{:02d}.pth'.format(opt.Model.name, epoch))
            print('Saving Snapshot: {}_best_{:02d}.pth'.format(opt.Model.name, epoch))
        else:
            logger.info('>>> not find the best epoch -> continue training ...')
            print('>>> not find the best epoch -> continue training ...')
    torch.save(model.state_dict(), save_path + '/' + '{}_{:02d}.pth'.format(opt.Model.name, epoch))
    logger.info('Saving Snapshot: {}_{:02d}.pth'.format(opt.Model.name, epoch))
    print('Saving Snapshot: {}_{:02d}.pth'.format(opt.Model.name, epoch))
    logger.info('Valid Best Epoch: {}, Metrics ({})'.format(best_epoch, best_score))
    print('Valid Best Epoch: {}, Metrics ({})'.format(best_epoch, best_score))
        
    
if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    opt = load_config(config_path) 
    train(opt)

