# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:21:58 2025

@author: FST
"""


import torch
#import torchvision
import timm
from torch.cuda.amp import autocast, GradScaler
import pandas as pd 
pd.set_option('display.width', 1000)  # 设置输出宽度足够大
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行
import os
from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
import random
import h5py
from functools import reduce
from io import BytesIO
from sklearn.metrics import fbeta_score
from PIL import Image
import time
import random
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead
import matplotlib.pyplot as plt
#import sklearn
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset, DataLoader, Sampler, SequentialSampler, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import  monai.transforms as transforms
from monai.utils import set_determinism

from einops import rearrange
from decoder import * #MyUnetDecoder3d,encode_for_resnet
from helper.dataset import *
from eval import *
from eval_cuda import *

import gc
from convert_3d import convert_3d
import warnings
warnings.filterwarnings("ignore")

os.makedirs("./model", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

model_list = os.listdir("./model/")
model_id = 0
for model_name in model_list:
    if int(model_name.split("_")[0])>model_id:
        model_id=int(model_name.split("_")[0].split(".")[0])
    
model_id += 1
print(f"model id:{model_id}")

class CFG:
    
    debug = 0
    # ============== model cfg =============
    in_chans = 3
    drop_rate = 0.
    drop_path_rate = 0.
    model_name = "Unet3D_encoder3D" #"Unet3D_encoder2_5D"  "Unet3D_encoder2D" "Unet3D_encoder3D" "Unet3D_encoder2D_3D"
    backbone = 'seresnext26d_32x4d' 
    freeze = 0
    
    # ============== training cfg =============
    extra_dataset = 0
    
    all_exp = ['TS_5_4','TS_6_4','TS_6_6','TS_69_2','TS_73_6','TS_86_3','TS_99_9'] 
    valid_exp = ['TS_73_6'] #'TS_73_6' TS_6_6 TS_5_4 TS_86_3
    train_exp = list(set(all_exp) - set(valid_exp))
    
    load = 183
    finetune = 0
    start_epoch = 0
    load_model_type = "best_cv" # [ best_loss, best_cv, last ] 
    
    train_batch_size = 3 #12
    valid_batch_size = 3 #12
    
    # ============== lr =============
    epochs = 50 # 30
    epochs = epochs*4 if extra_dataset else epochs
    patience = 20
    lr = 1e-5
    min_lr = 1e-6
    batch_scheduler = 0
    
    lr = 5e-5 if finetune else lr
    min_lr = lr if finetune else min_lr
    patience = 10 if finetune else patience
    
    # ============== loss ============= 
    loss_function = 3  # 1~7
    loss_weight =  [1,1,1,2,1,2,1]   
    weight = [1, 0, 2, 1, 2, 1]    
    
    # ============== aug =============
    mosaic_type = 8
    mosaic_prob = 0.5
    mixup_prob = 0.5
    cutmix_prob = 0.5
    
    W_H = 640
    img_size = 160  #160
    d = 64
    
    xy_stride = img_size//2
    z_stride = d//2
    
    train_aug_list = [
        transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),

        transforms.RandAffined(keys=["image", "mask"],
                               translate_range=[int(x*y) for x, y in zip([d,img_size,img_size], [0.3, 0.3, 0.3])],
                               rotate_range=(np.pi/2, np.pi/18, np.pi/18),
                               #rotate_range=(np.pi / 2, np.pi / 2, np.pi / 2),
                               scale_range=(0.1, 0.1, 0.1),
                               #padding_mode='zeros',
                               prob=0.8), #https://github.com/Project-MONAI/tutorials/blob/main/modules/3d_image_transforms.ipynb
        transforms.RandGridDistortiond(keys=("image", "mask"), prob=0.5, distort_limit=(-0.01, 0.01), mode="nearest"),
    ]
   
    valid_aug_list = []  

    # ============== path =============   
    load_smoothing_npy = "data/mask_npy/npy_smothing_1_0.5"
    pseudo = 0
    npy_folder = "mask_npy/npy_r0.1v10.0"
    exp_name = model_id
    outputs_path = './'
    model_dir = outputs_path + 'model/'
    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'
    
    # ============== other =============
    use_amp = True
    num_workers = 0
    seed = 69
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In[ ]:
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# In[ ]:

def init_logger(log_file):
    
    with open(log_file, 'w') as f:
        for attr in dir(CFG):
            if not callable(getattr(CFG, attr)) and not attr.startswith("__"):
                f.write('{} = {}\n'.format(attr, getattr(CFG, attr)))
                
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def set_seed(seed=42, cudnn_deterministic=True):
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = True #True False
    set_determinism(seed=seed)


# In[ ]:
def prepare_img_and_mask(img_npy_path,mask_npy_path):
    img = np.load(img_npy_path)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    D, H, W = img.shape
    img = np.pad(img, [[0, 0], [0, CFG.W_H - H], [0, CFG.W_H - W]], mode='constant', constant_values=0)
    img = np.stack([img]).astype(np.float32)/255.
    
    mask = np.load(mask_npy_path)
    D, H, W = mask.shape
    mask = np.pad(mask, [[0, 0], [0, CFG.W_H - H], [0, CFG.W_H - W]], mode='constant', constant_values=0)
    
    mask_per_class = []
    for i in range(7):
        m = np.zeros_like(mask)
        m[mask==i] = 255
        mask_per_class.append(m)
    mask = np.stack(mask_per_class).astype(np.float32)/255.

    return img,mask

def prepare_img(img_npy_path):
    img = np.load(img_npy_path)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    D, H, W = img.shape
    img = np.pad(img, [[0, 0], [0, CFG.W_H - H], [0, CFG.W_H - W]], mode='constant', constant_values=0)
    img = np.stack([img]).astype(np.float32)/255.
    
    return img

def prepare_dataset(exp_list,mode):
    
    imgs = []
    masks = []
    aux_mask = []
    
    bar = tqdm(enumerate(exp_list),total=len(exp_list))
    for idx,exp_name in bar:

        if mode=="train":
            
            dataset_list = ["denoised"]
            if CFG.extra_dataset:
                dataset_list = ["ctfdeconvolved","isonetcorrected","wbp","denoised"]
            
            for dataset in dataset_list:
                img_npy_path = f"data/img_npy/{dataset}/train_image_{exp_name}_{dataset}.npy"
                if CFG.load_smoothing_npy:
                    img = prepare_img(img_npy_path)
                    mask = np.load(f"{CFG.load_smoothing_npy}/train_label_{exp_name}.npy")
        
                else:
                    mask_npy_path = f"data/{CFG.npy_folder}/train_label_{exp_name}.npy"
                    img, mask = prepare_img_and_mask(img_npy_path,mask_npy_path)
                
                imgs.append(img)
                if dataset=="denoised":
                    masks.append(mask)
                    
        else:
            img_npy_path = f"data/img_npy/denoised/train_image_{exp_name}_denoised.npy"
            if CFG.load_smoothing_npy:
                img = prepare_img(img_npy_path)
                mask = np.load(f"{CFG.load_smoothing_npy}/train_label_{exp_name}.npy")
  
   
            else:
                mask_npy_path = f"data/{CFG.npy_folder}/train_label_{exp_name}.npy"
                img, mask = prepare_img_and_mask(img_npy_path,mask_npy_path)
            for x in list(range(0, CFG.W_H - CFG.img_size, CFG.xy_stride)) + [CFG.W_H - CFG.img_size]:
                for y in list(range(0, CFG.W_H - CFG.img_size, CFG.xy_stride)) + [CFG.W_H - CFG.img_size]:
                    for z in list(range(0, 184 - CFG.d, CFG.z_stride)) + [184 - CFG.d]:
                        im = img[:,z:z+CFG.d,y:y+CFG.img_size,x:x+CFG.img_size]
                        m = mask[:,z:z+CFG.d,y:y+CFG.img_size,x:x+CFG.img_size]
                        imgs.append(im)
                        masks.append(m)
                        
    
    if CFG.pseudo and mode=="train":  
        pseudo_list = os.listdir("data/extra_npy/")
        bar = tqdm(enumerate(pseudo_list),total=len(pseudo_list))
        for idx,exp_name in bar:
            
            img_npy_path = f"data/extra_npy/{exp_name}"
            mask_npy_path = f"data/pseudo_label/{CFG.pseudo}_best_cv/{exp_name}"
            img = prepare_img(img_npy_path)
            mask = np.load(mask_npy_path)
            
            for i in mask[0]:
                print(i.shape,np.max(i),np.min(i))
                plt.imshow(i)
                plt.show() 
                
            raise
            imgs.append(img)
            masks.append(mask)
           
    
    return {"imgs":imgs,
            "masks":masks}
 
def mosaic4(img_list, mask_list, img, mask):

    mosaic_img_list = []
    mosaic_mask_list = []
    
    r1 = random.randint(0,CFG.img_size)
    r2 = random.randint(0,CFG.img_size)
    wh = [(r1,r2),(r1,CFG.img_size-r2),(CFG.img_size-r1,r2),(CFG.img_size-r1,CFG.img_size-r2)]
    
    for i in range(4):
        if i!=0:
            idx = random.randint(0,len(img_list)-1)
            img = img_list[idx]
            mask = mask_list[idx]
        r = random.randint(0,184-CFG.d)
        r1 = random.randint(0,CFG.W_H-wh[i][0])
        r2 = random.randint(0,CFG.W_H-wh[i][1])
        im = img[:,r:r+CFG.d,r1:r1+wh[i][0],r2:r2+wh[i][1]]
        m = mask[:,r:r+CFG.d,r1:r1+wh[i][0],r2:r2+wh[i][1]]
        mosaic_img_list.append(im)
        mosaic_mask_list.append(m)

    #random.shuffle(mosaic_img_list)
    img1 = np.concatenate([mosaic_img_list[0], mosaic_img_list[1]], axis=3)  
    img2 = np.concatenate([mosaic_img_list[2], mosaic_img_list[3]], axis=3)  
    img = np.concatenate([img1, img2], axis=2)  
    
    #random.shuffle(mosaic_img_list)
    mask1 = np.concatenate([mosaic_mask_list[0], mosaic_mask_list[1]], axis=3)  
    mask2 = np.concatenate([mosaic_mask_list[2], mosaic_mask_list[3]], axis=3)  
    mask = np.concatenate([mask1, mask2], axis=2)  
    
    del img1, img2, mask1, mask2, mosaic_img_list, mosaic_mask_list
    gc.collect()
    
    return img, mask

def mosaic8(img_list, mask_list, img, mask):

    mosaic_img_list = []
    mosaic_mask_list = []
    
    r1 = random.randint(0,CFG.img_size)
    r2 = random.randint(0,CFG.img_size)
    d = random.randint(0,CFG.d)
    wh = [(r1,r2),(r1,CFG.img_size-r2),(CFG.img_size-r1,r2),(CFG.img_size-r1,CFG.img_size-r2),\
          (r1,r2),(r1,CFG.img_size-r2),(CFG.img_size-r1,r2),(CFG.img_size-r1,CFG.img_size-r2)]
    
    for i in range(8):
        if i!=0:
            idx = random.randint(0,len(img_list)-1)
            img = img_list[idx]
            mask = mask_list[idx]
        if i < 4:
            r = random.randint(0,184-d)
            r1 = random.randint(0,CFG.W_H-wh[i][0])
            r2 = random.randint(0,CFG.W_H-wh[i][1])
            im = img[:,r:r+d,r1:r1+wh[i][0],r2:r2+wh[i][1]]
            m = mask[:,r:r+d,r1:r1+wh[i][0],r2:r2+wh[i][1]]
        else:
            r = random.randint(0,184-(CFG.d-d))
            r1 = random.randint(0,CFG.W_H-wh[i][0])
            r2 = random.randint(0,CFG.W_H-wh[i][1])
            im = img[:,r:r+(CFG.d-d),r1:r1+wh[i][0],r2:r2+wh[i][1]]
            m = mask[:,r:r+(CFG.d-d),r1:r1+wh[i][0],r2:r2+wh[i][1]]
        mosaic_img_list.append(im)
        mosaic_mask_list.append(m)

    #random.shuffle(mosaic_img_list)
    img1 = np.concatenate([mosaic_img_list[0], mosaic_img_list[1]], axis=3)  
    img2 = np.concatenate([mosaic_img_list[2], mosaic_img_list[3]], axis=3)  
    img3 = np.concatenate([mosaic_img_list[4], mosaic_img_list[5]], axis=3)  
    img4 = np.concatenate([mosaic_img_list[6], mosaic_img_list[7]], axis=3)   
    img_top = np.concatenate([img1, img2], axis=2)  
    img_bot = np.concatenate([img3, img4], axis=2)  
    img =  np.concatenate([img_top, img_bot], axis=1)  

    #random.shuffle(mosaic_img_list)
    mask1 = np.concatenate([mosaic_mask_list[0], mosaic_mask_list[1]], axis=3)  
    mask2 = np.concatenate([mosaic_mask_list[2], mosaic_mask_list[3]], axis=3)  
    mask3 = np.concatenate([mosaic_mask_list[4], mosaic_mask_list[5]], axis=3)  
    mask4 = np.concatenate([mosaic_mask_list[6], mosaic_mask_list[7]], axis=3)  
    mask_top = np.concatenate([mask1, mask2], axis=2)  
    mask_bot = np.concatenate([mask3, mask4], axis=2)  
    mask = np.concatenate([mask_top, mask_bot], axis=1)  

    del img1, img2, img3, img4, img_top, img_bot, mosaic_img_list
    del mask1, mask2, mask3, mask4, mask_top, mask_bot, mosaic_mask_list
    gc.collect()
    
    return img, mask

def cutmix(img_list, mask_list, img, mask):

    w = random.randint(0,CFG.img_size//2)
    h = random.randint(0,CFG.img_size//2)
    d = random.randint(0,CFG.d//2)
    x1 = random.randint(0,w)
    y1 = random.randint(0,h)
    z1 = random.randint(0,d)
    x2 = random.randint(0,w)
    y2 = random.randint(0,h)
    z2 = random.randint(0,d)
    
    idx = random.randint(0,len(img_list)-1)
    img_cut = img_list[idx]
    mask_cut = mask_list[idx]
    img[:,z1:z1+d,y1:y1+h,x1:x1+w] = img_cut[:,z2:z2+d,y2:y2+h,x2:x2+w]
    mask[:,z1:z1+d,y1:y1+h,x1:x1+w] = mask_cut[:,z2:z2+d,y2:y2+h,x2:x2+w]
    
    del img_cut, mask_cut
    gc.collect()
    
    return img, mask
    
def get_transforms(data):
    if data == 'train':
        aug = transforms.Compose(CFG.train_aug_list)
    elif data == 'valid':
        aug = transforms.Compose(CFG.valid_aug_list)
    return aug

class CustomDataset(Dataset):
    def __init__(self, data, mode, transforms=None):
                       
        self.data = data
        self.mode = mode
        self.transforms = transforms
        
    def __len__(self):
        if self.mode=="train":

            xy_num = len(list(range(0, CFG.W_H - CFG.img_size, CFG.xy_stride))) + 1
            z_num = len(list(range(0, 184 - CFG.d, CFG.z_stride))) + 1
            num = xy_num*xy_num*z_num
            return len(self.data["imgs"])*num
        else:
            return len(self.data["imgs"])
        
    def __getitem__(self, idx):
        
        if self.mode=="train":
            img = self.data["imgs"][idx%len(self.data["imgs"])]
            mask = self.data["masks"][idx%len(self.data["masks"])]
            
            if random.random() < CFG.mosaic_prob:
                if CFG.mosaic_type==4:
                    img, mask = mosaic4(self.data["imgs"], self.data["masks"], img, mask)
                elif CFG.mosaic_type==8:
                    img, mask = mosaic8(self.data["imgs"], self.data["masks"], img, mask)
            else:       
                r = random.randint(0,184-CFG.d)
                r1 = random.randint(0,CFG.W_H-CFG.img_size)
                r2 = random.randint(0,CFG.W_H-CFG.img_size)
                img = img[:,r:r+CFG.d,r1:r1+CFG.img_size,r2:r2+CFG.img_size]
                mask = mask[:,r:r+CFG.d,r1:r1+CFG.img_size,r2:r2+CFG.img_size]
            if random.random() < CFG.cutmix_prob:
                img, mask = cutmix(self.data["imgs"], self.data["masks"], img, mask)
                
        else:
            img = self.data["imgs"][idx]
            mask = self.data["masks"][idx]
        data = self.transforms({'image':img, 'mask':mask})
        img = data['image']
        mask = data['mask']
        
        return {"img":img,
                "mask":mask,
                }


# In[ ]:
    
def prepare_loaders():
    
    print("loading train dataset ...")
    train_data = prepare_dataset(CFG.train_exp,mode="train")
    print("loading valid dataset ...")
    valid_data = prepare_dataset(CFG.valid_exp,mode="valid")

    train_dataset = CustomDataset(train_data, mode="train", transforms=get_transforms(data='train'))
    valid_dataset = CustomDataset(valid_data, mode="valid",transforms=get_transforms(data='valid'))
    
    train_loader =  DataLoader(train_dataset,
                              batch_size=CFG.train_batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=False)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=False)
    
    return train_loader, valid_loader

# In[ ]:

class Unet3D_encoder2D(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_class=6+1

        self.arch = CFG.backbone
        
        self.encoder_dim = {
                        'resnet18': [64, 64, 128, 256, 512, ],
                        'resnet18d': [64, 64, 128, 256, 512, ],
                        'resnet34d': [64, 64, 128, 256, 512, ],
                        'resnet50d': [64, 256, 512, 1024, 2048, ],
                        'resnext50_32x4d.a2_in1k': [64, 256, 512, 1024, 2048, ],
                        'seresnext26d_32x4d': [64, 256, 512, 1024, 2048],
                        'seresnext50_32x4d': [64, 256, 512, 1024, 2048],
                        'seresnext101_32x8d': [64, 256, 512, 1024, 2048],
                        'convnext_small.fb_in22k': [96, 192, 384, 768],
                        'convnext_tiny.fb_in22k': [96, 192, 384, 768],
                        'convnext_tiny.in12k_ft_in1k': [96, 192, 384, 768],
                        'convnext_base.fb_in22k': [128, 256, 512, 1024],
                        'coatnet_rmlp_1_rw2_224.sw_in12k': [64, 96, 192, 384, 768],
                        'tf_efficientnetv2_s.in21k':[24, 48, 64, 160, 256],
                        'tf_efficientnetv2_m.in21k':[24, 48, 80, 176, 512],
                        'tf_efficientnetv2_l.in21k':[32, 64, 96, 224, 640],
                        'tf_efficientnet_b4.ns_jft_in1k':[32, 56, 160, 448],
                        'tf_efficientnet_b5.ns_jft_in1k':[40, 64, 176, 512],
                        'tf_efficientnet_b6.ns_jft_in1k':[40, 72, 200, 576],
                        'tf_efficientnet_b7.ns_jft_in1k':[48, 80, 224, 640],
                        'regnety_016': [32, 48, 120, 336, 888],
                        'pvt_v2_b1': [64, 128, 320, 512],
                        'pvt_v2_b2': [64, 128, 320, 512],
                        'pvt_v2_b4': [64, 128, 320, 512],
                        }.get(self.arch, [768])
        
        #self.decoder_dim = [384, 192, 96, 32, 16]
        self.decoder_dim = [384, 192, 96, 32, 16]

        self.encoder = timm.create_model(
                                            model_name=self.arch,
                                            pretrained=True,
                                            in_chans=3,
                                            num_classes=0,
                                            global_pool='',
                                            features_only=True,
                                        )
        
        self.decoder = MyUnetDecoder3d(
            in_channel=self.encoder_dim[-1],
            skip_channel=self.encoder_dim[:-1][::-1]+[0],
            out_channel=self.decoder_dim,
        )
        
        self.segmentation_head = nn.Conv3d(self.decoder_dim[:len(self.encoder_dim)][-1],self.num_class, kernel_size=1)
        self.L = ['logits']
        
        
        self.depth_scaling = [2,2,2,2,1]

        self.depth_dim = [[CFG.d,CFG.d//2],
                          [CFG.d//2,CFG.d//4],
                          [CFG.d//4,CFG.d//8],
                          [CFG.d//8,CFG.d//16],
                          [CFG.d//16,CFG.d//16]]
        
        if CFG.freeze:
            for p in self.encoder.parameters():
               p.requires_grad = False
               
            for p in self.decoder.parameters():
               p.requires_grad = False
                      
    def forward(self, x):

        B, C, D, H, W = x.shape
        x = x.reshape(B*D, C, H, W)
        x = x.expand(-1, 3, -1, -1)
 
        #encode = self.encoder(x)[-5:]
        if "resne" in self.arch:
            encode = encode_for_resnet(self.encoder, x, B, depth_scaling=self.depth_scaling)
            #encode = encode_for_resnet_v2(self.encoder, self.weight, x, B, depth_scaling=self.depth_scaling)
        elif "efficient" in self.arch:
            if  "tf_efficientnetv2_s" in self.arch:
                encode = encode_for_effnetv2s(self.encoder, x, B, depth_scaling=self.depth_scaling)
            else:
                encode = encode_for_effnet(self.encoder, x, B, depth_scaling=self.depth_scaling)
        elif "coat" in self.arch:
            encode = encode_for_coatet(self.encoder, x, B, depth_scaling=self.depth_scaling)
        elif "regnety" in self.arch:

            encode = encode_for_regnety(self.encoder, x, B, depth_scaling=self.depth_scaling)
            
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1]+[None], depth_scaling=self.depth_scaling[::-1]
        )

        logit = self.segmentation_head(last)
        
        #print('logit', logit.shape)
        
        return {'logits':logit}
   
    def criterion_v3(self,outputs,data):
        
        loss_list = []
        for L in self.L:
            logit = outputs[L]
            mask = data['mask']
            lam = data["lam"]
            mask_shuffled = data["mask_shuffled"] if lam is not None else None
            
            if len(CFG.loss_weight)!=0:
                CELoss = nn.CrossEntropyLoss(weight=torch.tensor(CFG.loss_weight).to(CFG.device),reduction="none")
                
                l = CELoss(logit,mask)

                B, C, D, H, W = mask.shape
                
                pixel_weight = torch.ones((B, D, H, W)).to(CFG.device)
                loss_weight = CFG.loss_weight
                for idx,i in enumerate(loss_weight):
                    if i!=1:
                        pixel_weight[mask[:,idx]==1] = i
     
                loss = torch.sum(pixel_weight*l)/torch.sum(pixel_weight)
                if lam is not None:
                    l = CELoss(logit,mask_shuffled)
                    B, C, D, H, W = mask_shuffled.shape
                    pixel_weight = torch.ones((B, D, H, W)).to(CFG.device)
                    loss_weight = CFG.loss_weight
                    for idx,i in enumerate(loss_weight):
                        if i!=1:
                            pixel_weight[mask_shuffled[:,idx]==1] = i
                    loss1 = torch.sum(pixel_weight*l)/torch.sum(pixel_weight)
                    
                    loss = loss*lam + loss1*(1-lam)
                    
            else:
                CELoss = nn.CrossEntropyLoss()
                loss = CELoss(logit,mask)
                if lam is not None:
                    loss1 = CELoss(logit,mask_shuffled)
                    loss = loss*lam + loss1*(1-lam)
            loss_list.append(loss)
        
        return sum(loss_list)/len(loss_list)
    
    def criterion_v2(self,outputs,data):
        
        loss_list = []
        for L in self.L:
            logit = outputs[L]
            mask = data['mask']
            lam = data["lam"]
            mask_shuffled = data["mask_shuffled"] if lam is not None else None
            
            CELoss = nn.CrossEntropyLoss(weight=torch.tensor(CFG.loss_weight).to(CFG.device),reduction="mean")
            loss = CELoss(logit,mask)

            if lam is not None:
                loss1 = CELoss(logit,mask_shuffled)
                loss = loss*lam + loss1*(1-lam)
                    
            
            loss_list.append(loss)
        
        return sum(loss_list)/len(loss_list)
    
    def criterion_v1(self,outputs,data):
        
        loss_list = []
        for L in self.L:
            logit = outputs[L]
            mask = data['mask']
            lam = data["lam"]
            mask_shuffled = data["mask_shuffled"] if lam is not None else None
            
            if len(CFG.loss_weight)!=0:
                CELoss = nn.CrossEntropyLoss(reduction="none")
                
                l = CELoss(logit,mask)

                B, C, D, H, W = mask.shape
                
                pixel_weight = torch.ones((B, D, H, W)).to(CFG.device)
                loss_weight = CFG.loss_weight
                for idx,i in enumerate(loss_weight):
                    if i!=1:
                        pixel_weight[mask[:,idx]==1] = i
     
                loss = torch.sum(pixel_weight*l)/torch.sum(pixel_weight)
                if lam is not None:
                    l = CELoss(logit,mask_shuffled)
                    B, C, D, H, W = mask_shuffled.shape
                    pixel_weight = torch.ones((B, D, H, W)).to(CFG.device)
                    loss_weight = CFG.loss_weight
                    for idx,i in enumerate(loss_weight):
                        if i!=1:
                            pixel_weight[mask_shuffled[:,idx]==1] = i
                    loss1 = torch.sum(pixel_weight*l)/torch.sum(pixel_weight)
                    
                    loss = loss*lam + loss1*(1-lam)
                    
            else:
                CELoss = nn.CrossEntropyLoss()
                loss = CELoss(logit,mask)
                if lam is not None:
                    loss1 = CELoss(logit,mask_shuffled)
                    loss = loss*lam + loss1*(1-lam)
            loss_list.append(loss)
        
        return sum(loss_list)/len(loss_list)
    
class Unet3D_encoder3D(Unet3D_encoder2D):
    def __init__(self):
        super().__init__()
        
        num_class=6+1
        
        self.encoder = timm.create_model(
            CFG.backbone,
            in_chans=CFG.in_chans,
            features_only=True,
            drop_rate=CFG.drop_rate,
            drop_path_rate=CFG.drop_path_rate,
            pretrained=True
        )
        self.n_blocks = 4
        g = self.encoder(torch.rand(1, CFG.in_chans, CFG.img_size, CFG.img_size))
        
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels[:self.n_blocks+1],
            decoder_channels=decoder_channels[:self.n_blocks],
            n_blocks=self.n_blocks,
        )
        self.segmentation_head = nn.Conv2d(decoder_channels[self.n_blocks-1], num_class, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self,x):
        B, C, D, H, W = x.shape
        x = x.expand(-1, 3, -1, -1, -1)
        
        global_features = [0] + self.encoder(x)[:self.n_blocks]
        seg_features = self.decoder(*global_features)
        logit = self.segmentation_head(seg_features)
        
        return {"logits":logit}

def build_model(): 
    
    if CFG.model_name=="Unet3D_encoder2D":
        model = Unet3D_encoder2D()
    elif CFG.model_name=="Unet3D_encoder2_5D":
        model = Unet3D_encoder2_5D()
    elif CFG.model_name=="Unet3D_encoder3D":
        model = Unet3D_encoder3D()
        model = convert_3d(model)
  
    model = nn.DataParallel(model)
    
    return model

# In[ ]:
def mixup(data):
    
    input = data["img"]
    truth = data["mask"]
    
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(0, 1)
    input = input * lam + shuffled_input * (1 - lam)
    
    data["img"] = input
    data["mask_shuffled"] = shuffled_labels
    data["lam"] = lam
    return data

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def train_fn(train_loader, model, epoch, optimizer, scheduler):
    model.train()
    #scaler = GradScaler(enabled=CFG.use_amp)

    scaler = torch.amp.GradScaler('cuda')
    losses = AverageMeter()
    
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, data in bar:
        data = batch_to_device(data,CFG.device)
        
        data["lam"] = None
        if random.random() < CFG.mixup_prob:
            data = mixup(data)
            
        imgs = data["img"]
        
        batch_size = imgs.size(0)
        optimizer.zero_grad()
       
        #with autocast(CFG.use_amp):
        with torch.amp.autocast('cuda'):
  
            outputs = model(imgs)
            
            if CFG.loss_function==1:
                loss = model.module.criterion_v1(outputs,data)
            elif CFG.loss_function==2:
                loss = model.module.criterion_v2(outputs,data)
            elif CFG.loss_function==3:
                loss = model.module.criterion_v3(outputs,data)
        
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
                
        scaler.step(optimizer)
        scaler.update()
        
        bar.set_postfix(Epoch=epoch, 
                        Lr=optimizer.param_groups[0]['lr'], 
                        Train_Loss=losses.avg,
                        )
        
        if CFG.batch_scheduler:
            scheduler.step()
    
    bar.close()

    torch.cuda.empty_cache()
    
    del data, outputs
    gc.collect()
    
    return losses.avg

def valid_fn(valid_loader, model, epoch):
        
    y_pred = []
    
    model.eval()
    losses = AverageMeter()
    
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for step, data in bar:
        
        data = batch_to_device(data,CFG.device)
        
        data["lam"] = None
        imgs = data["img"]
      
        batch_size = imgs.size(0)
        
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
    
                outputs = model(imgs)
                logits = outputs["logits"]
                
                if CFG.loss_function==1:
                    loss = model.module.criterion_v1(outputs,data)
                elif CFG.loss_function==2:
                    loss = model.module.criterion_v2(outputs,data)
                elif CFG.loss_function==3:
                    loss = model.module.criterion_v3(outputs,data)
         
        losses.update(loss.item(), batch_size)
                
        bar.set_postfix(Epoch=epoch,
                        Valid_Loss=losses.avg,
                        )
        logits = F.softmax(logits,1).cpu().detach().numpy()
        y_pred.append(logits)
        
    del data, outputs, logits
    gc.collect()
    
    y_pred = np.concatenate(y_pred,0)
            
    #volume = read_one_data(CFG.valid_exp[0], static_dir=f'data/czii-cryo-et-object-identification/train/static/ExperimentRuns')
    D, H, W = 184, 630, 630 #volume.shape
    probability = np.zeros((7, 184, CFG.W_H, CFG.W_H), dtype=np.float32)
    count = np.zeros((7, 184, CFG.W_H, CFG.W_H), dtype=np.float32)
    index = 0
    for x in list(range(0, CFG.W_H - CFG.img_size, CFG.xy_stride)) + [CFG.W_H - CFG.img_size]:
        for y in list(range(0, CFG.W_H - CFG.img_size, CFG.xy_stride)) + [CFG.W_H - CFG.img_size]:
            for z in list(range(0, 184 - CFG.d, CFG.z_stride)) + [184 - CFG.d]:
                probability[:, z:z+CFG.d,y:y+CFG.img_size,x:x+CFG.img_size] += y_pred[index]
                count[:, z:z+CFG.d,y:y+CFG.img_size,x:x+CFG.img_size] += 1
                index += 1
                
    del y_pred
    gc.collect()
    torch.cuda.empty_cache()
    
    probability = probability / (count + 0.0001)
    probability = probability[:,:D,:H,:W]
    #probability = torch.from_numpy(probability).to(CFG.device)
    
    result = calculate_cv(probability)
    
    bar.set_postfix(Epoch=epoch,
                    Valid_Loss=losses.avg,
                    CV=result["lb_score"])
    bar.close()
    
    
    torch.cuda.empty_cache()
    del probability, count
    gc.collect()
    
    return losses.avg, result

# In[ ]:
def calculate_cv(probability):
    
    weight = np.array(CFG.weight)
    f_beta4 = []
    gbs = []
    
    bar = tqdm(enumerate(PARTICLE),total=len(PARTICLE))
    for idx,p in bar:
        best_f_beta4 = 0
        for threshold in range(5,100,5):
            threshold = threshold/100
            location = probability_to_location_per_class(probability,threshold,p)
           
            df = location_to_df_per_class(location,p)
            df.insert(loc=0, column='experiment', value=CFG.valid_exp[0])
            df.insert(loc=0, column='id', value=np.arange(len(df)))
            gb, score = compute_lb_per_class(df,f'data/czii-cryo-et-object-identification/train/overlay/ExperimentRuns',p)
            gb["threshold"] = threshold

            if score.item()>=best_f_beta4:
                best_f_beta4 = score.item()
                best_gb = gb
                
        f_beta4.append(best_f_beta4)
        gbs.append(best_gb)
 
    gbs = pd.concat(gbs)
    gbs["weight"] = weight
    f_beta4 = np.array(f_beta4)
    lb_score = (f_beta4 * weight).sum() / weight.sum()
   
    return {"gb":gbs,
            "lb_score":lb_score
            }

# In[ ]:  

if __name__ == "__main__":    
    
    set_seed(CFG.seed)
    Logger = init_logger(CFG.log_path)
    Logger.info('=============================================================')  
    Logger.info(f'train exp: {CFG.train_exp}')  
    Logger.info(f'valid exp: {CFG.valid_exp}')  
    Logger.info('=============================================================') 
 
    train_loader, valid_loader = prepare_loaders() 
    
    model = build_model()
    if CFG.load:
        print(f'loading {CFG.load}_{CFG.load_model_type}.pth')
        state = torch.load(CFG.model_dir + f'{CFG.load}_{CFG.load_model_type}.pth')
        model.load_state_dict(state, strict=False)
    model.to(CFG.device)
    
   
    
    #optimizer = AdamW(model.parameters(), lr=CFG.lr)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, CFG.epochs, eta_min=CFG.min_lr)
    
    best_score = -1
    best_loss = np.inf
    start_epoch = 0 if not CFG.load else CFG.start_epoch            
    early_stop = 0
    
    for epoch in range(CFG.epochs):
        
        if start_epoch<=epoch:
            
            Logger.info("-"*80)
            
            # train
            start_time = time.time()
            avg_loss = 0
            avg_loss = train_fn(train_loader, model, epoch, optimizer, scheduler)

            elapsed = time.time() - start_time
            Logger.info( f'Epoch {epoch} / lr: {optimizer.param_groups[0]["lr"]:.6f} / train_loss: {avg_loss:.6f} / time: {elapsed:.0f}s')
            
            if not CFG.batch_scheduler:
                scheduler.step()
                
            # eval
            start_time = time.time()
            avg_val_loss, result = valid_fn(valid_loader, model, epoch)
            cv = result["lb_score"]
            elapsed = time.time() - start_time
            Logger.info( f'Epoch {epoch} / CV: {cv:.6f} / valid_loss: {avg_val_loss:.6f} / time: {elapsed:.0f}s')
            Logger.info(result["gb"])
            Logger.info(list(result["gb"]["threshold"]))
            Logger.info(list(result["gb"]["f-beta4"]))

            if not CFG.debug:
 
                torch.save( model.state_dict(),CFG.model_dir + f'{CFG.exp_name}_last.pth')
                
                if cv > best_score:
                    best_score = cv
                    torch.save(model.state_dict(), CFG.model_dir + f'{CFG.exp_name}_best_cv.pth')
                    Logger.info(f'*************** - Save Best CV Model: {best_score:.6f} ')
                    early_stop = 0
                else:
                    early_stop+=1   
               
                    
                if early_stop>=CFG.patience:
                    break
              
        else:
            scheduler.step()
            
    CFG.exp_name+=1   
    Logger.info('***************')  
    Logger.info(f'Best CV: {best_score:.6f} ')
    Logger.info('***************') 
    Logger.info('') 
    
    print(f"model id:{model_id}")    
        

    
    
    

