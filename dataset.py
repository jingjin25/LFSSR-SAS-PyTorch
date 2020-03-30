import torch.utils.data as data
import torch
import h5py
import numpy as np
import random
import cv2
from scipy import misc

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, scale, patch_size):
        super(DatasetFromHdf5, self).__init__()
        
        hf = h5py.File(file_path)
        self.img_HR = hf.get('img_HR')           # [N,ah,aw,h,w]
        self.img_LR_2 = hf.get('img_LR_2')   # [N,ah,aw,h/2,w/2]
        self.img_LR_4 = hf.get('img_LR_4')   # [N,ah,aw,h/4,w/4]
        
        self.img_size = hf.get('img_size') #[N,2]
        
        self.scale = scale        
        self.psize = patch_size
    
    def __getitem__(self, index):
                        
        # get one item
        hr = self.img_HR[index]       # [ah,aw,h,w]
        lr_2 = self.img_LR_2[index]   # [ah,aw,h/2,w/2]
        lr_4 = self.img_LR_4[index]   # [ah,aw,h/4,w/4]
                                               
        # crop to patch
        H, W = self.img_size[index]
        x = random.randrange(0, H-self.psize, 8)    
        y = random.randrange(0, W-self.psize, 8) 
        hr = hr[:, :, x:x+self.psize, y:y+self.psize] # [ah,aw,ph,pw]
        lr_2 = lr_2[:, :, x//2:x//2+self.psize//2, y//2:y//2+self.psize//2] # [ah,aw,ph/2,pw/2]
        lr_4 = lr_4[:, :, x//4:x//4+self.psize//4, y//4:y//4+self.psize//4] # [ah,aw,ph/4,pw/4]  

        # 4D augmentation
        # flip
        if np.random.rand(1)>0.5:
            hr = np.flip(np.flip(hr,0),2)
            lr_2 = np.flip(np.flip(lr_2,0),2)
            lr_4 = np.flip(np.flip(lr_4,0),2)  
            # lr_8 = np.flip(np.flip(lr_8,0),2)                
        if np.random.rand(1)>0.5:
            hr = np.flip(np.flip(hr,1),3)
            lr_2 = np.flip(np.flip(lr_2,1),3)
            lr_4 = np.flip(np.flip(lr_4,1),3) 
            # lr_8 = np.flip(np.flip(lr_8,1),3)
        # rotate
        r_ang = np.random.randint(1,5)
        hr = np.rot90(hr,r_ang,(2,3))
        hr = np.rot90(hr,r_ang,(0,1))
        lr_2 = np.rot90(lr_2,r_ang,(2,3))
        lr_2 = np.rot90(lr_2,r_ang,(0,1))           
        lr_4 = np.rot90(lr_4,r_ang,(2,3))
        lr_4 = np.rot90(lr_4,r_ang,(0,1)) 

        # to tensor     
        hr = hr.reshape(-1,self.psize,self.psize) # [an,ph,pw]
        lr_2 = lr_2.reshape(-1,self.psize//2,self.psize//2) #[an,phs,pws]
        lr_4 = lr_4.reshape(-1,self.psize//4,self.psize//4) # [an,phs,pws]

        hr = torch.from_numpy(hr.astype(np.float32)/255.0)
        lr_2 = torch.from_numpy(lr_2.astype(np.float32)/255.0)  
        lr_4 = torch.from_numpy(lr_4.astype(np.float32)/255.0)
        return hr, lr_2, lr_4

    def __len__(self):
        return self.img_HR.shape[0]