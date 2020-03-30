import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
from os.path import join

import math
import copy
import pandas as pd
import time 

import h5py
import matplotlib
matplotlib.use('Agg')
from scipy import misc
from skimage.measure import compare_ssim  

from model import net2x, net4x

#--------------------------------------------------------------------------#
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#----------------------------------------------------------------------------------#
# Test settings
parser = argparse.ArgumentParser(description="PyTorch LFSSR-SAS demo")
parser.add_argument("--model_path", type=str, default="pretrained_models/model_2x.pth", help="model dir")
parser.add_argument("--scale", type=int, default=2, help="SR factor")
parser.add_argument("--test_dataset", type=str, default="", help="dataset for test")
parser.add_argument("--angular_num", type=int, default=7, help="Size of one angular dim")
parser.add_argument("--layer_num", type=int, default=6, help="number of SAS layers")
parser.add_argument("--save_img", type=int, default=1, help="save image or not")

opt = parser.parse_args()
print(opt)
#-----------------------------------------------------------------------------------#   
class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, scale):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
               
        self.GT_y = hf.get('/GT_y')      #[N,aw,ah,h,w]
        self.LR_ycbcr = hf.get('/LR_ycbcr') #[N,ah,aw,3,h/s,w/s]

        self.scale = scale

    def __getitem__(self, index):

        h = self.GT_y.shape[3]
        w = self.GT_y.shape[4]
        
        gt_y = self.GT_y[index]
        gt_y = gt_y.reshape(-1, h, w)
        gt_y = torch.from_numpy(gt_y.astype(np.float32)/255.0)

        lr_ycbcr = self.LR_ycbcr[index]
        lr_ycbcr = torch.from_numpy(lr_ycbcr.astype(np.float32)/255.0)       

        lr_y = lr_ycbcr[:, :, 0, :, :].clone().view(-1, h//self.scale, w//self.scale)
        
        lr_ycbcr_up = lr_ycbcr.view(1, -1, h//self.scale, w//self.scale)
        lr_ycbcr_up = torch.nn.functional.interpolate(lr_ycbcr_up, scale_factor=self.scale, mode='bicubic',align_corners=False)
        lr_ycbcr_up = lr_ycbcr_up.view(-1, 3, h, w)
        
        return gt_y, lr_ycbcr_up, lr_y 
        
    def __len__(self):
        return self.GT_y.shape[0]
#-----------------------------------------------------------------------------------#        
#-------------------------------------------------------------------------------#

if opt.save_img:
    save_dir = 'results/saveImg/{}_x{}'.format(opt.test_dataset, opt.scale)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

an = opt.angular_num
#------------------------------------------------------------------------#
# Data loader
print('===> Loading test datasets')
data_path = join('LFData', 'test_{}_x{}.h5'.format(opt.test_dataset,opt.scale))
test_set = DatasetFromHdf5(data_path,opt.scale)
test_loader = DataLoader(dataset=test_set,batch_size=1,shuffle=False)
print('loaded {} LFIs from {}'.format(len(test_loader), data_path))
#-------------------------------------------------------------------------#
# Build model
print("===> building network")
srnet_name = 'net{}x'.format(opt.scale)
model = eval(srnet_name)(an,opt.layer_num).to(device)
#------------------------------------------------------------------------#

#-------------------------------------------------------------------------#    
# test  
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16. / 255.
    rgb[:,1:] -= 128. / 255.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 1).reshape(shape).astype(np.float32)

def compt_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
def test():
    model.eval()
    csv_name = 'results/res_{}_x{}_{}x{}_L{}.csv'.format(opt.test_dataset, opt.scale, opt.angular_num, opt.angular_num, opt.layer_num)
    
    lf_list = []
    lf_psnr_y_list = []
    lf_ssim_y_list = []
    
    with torch.no_grad():
        for k, batch in enumerate(test_loader):
            print('testing LF {}{}'.format(opt.test_dataset, k))
            #----------- SR ---------------#
            gt_y, sr_ycbcr, lr_y  = batch[0].numpy(),batch[1].numpy(),batch[2]            

            start = time.time()
            lr_y = lr_y.to(device)
            sr_list = model(lr_y)
            sr_y = sr_list[0].cpu()
            end = time.time()
            print('running time: ',end-start)
             
            sr_y = sr_y.numpy()          
            sr_ycbcr[:, :, 0] = sr_y
            #---------compute average PSNR/SSIM for this LFI----------#
            
            view_list = []
            view_psnr_y_list = []
            view_ssim_y_list = []
          
            for i in range(an*an):
                if opt.save_img:
                    img_name = '{}/SR{}_view{}.png'.format(save_dir, k, i)
                    sr_rgb_temp = ycbcr2rgb(np.transpose(sr_ycbcr[0, i], (1, 2, 0)))
                    img = (sr_rgb_temp.clip(0, 1)*255.0).astype(np.uint8)
                    misc.toimage(img, cmin=0, cmax=255).save(img_name)
  
                cur_psnr = compt_psnr(gt_y[0, i], sr_y[0, i])
                cur_ssim = compare_ssim((gt_y[0, i]*255.0).astype(np.uint8), (sr_y[0, i]*255.0).astype(np.uint8), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

                view_list.append(i)
                view_psnr_y_list.append(cur_psnr)
                view_ssim_y_list.append(cur_ssim)

            dataframe_lfi = pd.DataFrame({'View_LFI{}'.format(k): view_list, 'psnr Y':view_psnr_y_list,'ssim Y':view_ssim_y_list})
            dataframe_lfi.to_csv(csv_name, index=False, sep=',', mode='a')

            lf_list.append(k)
            lf_psnr_y_list.append(np.mean(view_psnr_y_list))
            lf_ssim_y_list.append(np.mean(view_ssim_y_list))

            print('Avg. Y PSNR: {:.2f}; Avg. Y SSIM: {:.3f}'.format(np.mean(view_psnr_y_list), np.mean(view_ssim_y_list)))

    dataframe_lfi = pd.DataFrame({'lfiNo': lf_list, 'psnr Y':lf_psnr_y_list, 'ssim Y':lf_ssim_y_list})
    dataframe_lfi.to_csv(csv_name, index = False, sep=',', mode='a')
    dataframe_lfi = pd.DataFrame({'summary': ['avg'], 'psnr Y':[np.mean(lf_psnr_y_list)], 'ssim Y':[np.mean(lf_ssim_y_list)]})                  
    dataframe_lfi.to_csv(csv_name, index=False, sep=',', mode='a')

    print('Over all {} LFIs on {}: Avg. Y PSNR: {:.2f}, Avg. Y SSIM: {:.3f}'.format(len(test_loader), opt.test_dataset, np.mean(lf_psnr_y_list), np.mean(lf_ssim_y_list)))
#------------------------------------------------------------------------#

print('===> testing pretrained model')
checkpoint = torch.load(opt.model_path)
model.load_state_dict(checkpoint['model'])        
print('loaded model {}'.format(opt.model_path))
test()
