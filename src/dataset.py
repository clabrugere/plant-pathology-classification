import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import cv2
from albumentations import Compose, Flip, RandomScale, ShiftScaleRotate, RandomBrightnessContrast, Rotate, RandomCrop, CenterCrop, Resize, Blur, CLAHE, Equalize, Normalize, OneOf, IAASharpen, IAAEmboss
from sklearn.model_selection import train_test_split
from . import config


class PlantPathology(Dataset):
    
    def __init__(self, df, label_cols=None, is_test=False, apply_transforms=True, to_tensor=True):
        
        self.is_test = is_test
        self.apply_transforms = apply_transforms
        self.to_tensor = to_tensor
        
        # load metadata
        self.df_metadata = df
        self.image_ids = self.df_metadata['image_id'].values
        
        if not self.is_test:
            self.label_cols = label_cols
            self.labels = self.df_metadata[self.label_cols].values
            
            # class weights
            self.label_weights = np.log(self.labels.shape[0] / self.labels.sum(axis=0) - 1)
        
        self.transforms = Compose([
            Flip(p=0.8),
            ShiftScaleRotate(shift_limit=0.05,
                             scale_limit=0.2,
                             rotate_limit=90,
                             p=1.),
            RandomBrightnessContrast(p=1.),
            OneOf([
                IAASharpen(),
                IAAEmboss(),
            ], p=0.5),
            RandomCrop(1024, 1024, p=1.),
            Resize(64, 64),
            CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.),
        ])
    
    def __len__(self):
        return self.image_ids.shape[0]
    
    def __getitem__(self, index):
        # read file
        if torch.is_tensor(index):
            index = index.tolist()
        
        image_path = config.get_image_filename(self.image_ids[index])
        image = cv2.imread(image_path)
        
        if self.apply_transforms:
            image = self._transform(image)
        
        if self.to_tensor:
            image = self._to_tensor(image)
        
        if self.is_test:
            return image
        else:
            label = self.labels[index]
            return (image, label)
    
    def _transform(self, image):
        return self.transforms(image=image)['image']
    
    def _to_tensor(self, image):
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        
        return image
    
    def label_from_vect(self, label_vector):
        return self.label_cols[np.argmax(label_vector)]


def stratified_split(df, label_cols, test_size=.2, shuffle=True):
    '''split a dataframe into a training and validation while preserving classes distributions
    '''
    train, test, _, _ = train_test_split(df, df[label_cols], test_size=test_size, stratify=df[label_cols], shuffle=True)
    
    return train, test
    

    
def oversample(df, label_cols, factor, balance_classes=True):
    '''duplicate samples in a dataframe according to the classes distributions and a multiplying factor
    '''
    if balance_classes:
        class_balance = df[label_cols].sum(axis=0) / df[label_cols].shape[0]
        class_balance = np.round(class_balance.max() / class_balance).astype('int8').to_dict()
    else:
        class_balance = {k: 1 for k in label_cols}
    
    for k, v in class_balance.items():
        df = df.append([df[df[k] == 1]]*factor*v, ignore_index=True)
    
    return df