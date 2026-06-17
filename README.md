# Frequency Separation and Aggregation-induced Contrastive Learning for Ultrasound Thyroid Nodule Segmentation

## Data
The DDTI, TN3K, and TUCC are publicly available datasets, they differ considerably in acquisition devices, imaging protocols, and patient populations. Therefore, we integrate them under a unified preprocessing and evaluation framework to establish a benchmark protocol for thyroid ultrasound segmentation. This benchmark enables comprehensive assessment of both segmentation accuracy and cross-dataset generalization.
### Downloading data
- Download the ultrasound datasets [DDTI](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images), [TN3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation),  and [TUCC](https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5).

### Dataset Organization
```text
TNS/
├── TrainDataset/       # Training set
│   └── file_list.txt/  # File name list
├── ValidDataset/       # Validation set
│   ├── Imgs/          
│   └── file_list.txt/           # Validation masks
└── TestDataset/        # Test set
    ├── DDTI/           # DDTI test set
    │   ├── Imgs/
    │   └── file_list.txt/
    ├── TN3K/           # TN3K test set
    │   ├── Imgs/
    │   └── file_list.txt/
    └── TUCC/           # TUCC test set
        ├── Imgs/
        └── file_list.txt/
```

## Training
The method and training code have been uploaded, and we will continue to optimize them.

## Prediction maps for all models can be found from [Google Drive](https://drive.google.com/file/d/1Jvbm1jpWOSxUFUhcGStxsKf0WTtqB3X4/view?usp=sharing)
