
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import math
import os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset  import DatasetFromHdf5
from model  import net2x, net4x
#--------------------------------------------------------------------------#
# Training settings
parser = argparse.ArgumentParser(description="PyTorch LFSSR-SAS training")

#training settings
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=250, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default = 64, help="Training patch size")
parser.add_argument("--batch_size", type=int, default = 1, help="Training batch size")
parser.add_argument("--resume_epoch", type=int, default=0, help="resume from checkpoint epoch")
parser.add_argument("--num_cp", type=int, default=5, help="Number of epoches for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epoches for saving loss figure")
#dataset
parser.add_argument("--dataset", type=str, default="all", help="Dataset for training")
parser.add_argument("--angular_num", type=int, default=7, help="Size of one angular dim")
#model 
parser.add_argument("--scale", type=int, default=2, help="SR factor")
parser.add_argument("--layer_num", type=int, default=6, help="number of SAS layers")

opt = parser.parse_args()
print(opt)
#--------------------------------------------------------------------------#
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#--------------------------------------------------------------------------#
torch.manual_seed(1)
model_dir = 'model_x{}_{}{}x{}_lr{}_step{}x{}_l{}'.format(opt.scale, opt.dataset, opt.angular_num, opt.angular_num, opt.lr, opt.step, opt.reduce, opt.layer_num)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
an = opt.angular_num
#--------------------------------------------------------------------------#
# Data loader
print('===> Loading datasets')
dataset_path = os.path.join('LFData', 'train_{}.h5'.format(opt.dataset))
train_set = DatasetFromHdf5(dataset_path, opt.scale, opt.patch_size)
train_loader = DataLoader(dataset=train_set,batch_size=opt.batch_size,shuffle=True)
print('loaded {} LFIs from {}'.format(len(train_loader),dataset_path))
#--------------------------------------------------------------------------#
# Build model
print("===> building network")
srnet_name = 'net{}x'.format(opt.scale)
model = eval(srnet_name)(an, opt.layer_num).to(device)
#-------------------------------------------------------------------------#
# optimizer and loss logger
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
losslogger = defaultdict(list)

#------------------------------------------------------------------------#    
# optionally resume from a checkpoint
if opt.resume_epoch:
    resume_path = join(model_dir,'model_epoch_{}.pth'.format(opt.resume_epoch))
    if os.path.isfile(resume_path):
        print("==> loading checkpoint 'epoch{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        losslogger = checkpoint['losslogger']
    else:
        print("==> no model found at 'epoch{}'".format(opt.resume_epoch))
#------------------------------------------------------------------------#
# loss
def L1_Charbonnier_loss(X,Y):
        eps = 1e-6
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + eps )
        loss = torch.sum(error) / torch.numel(error)
        return loss
#-----------------------------------------------------------------------#  
 
def train(epoch): 
    model.train()
    scheduler.step()
    
    loss_count = 0.
    
    for k in range(50):  
        for i, batch in enumerate(train_loader, 1):

            lr = batch[int(math.log(opt.scale, 2))].to(device)
            hr_list = batch[0:int(math.log(opt.scale, 2))]
            
            sr_list = model(lr)
            
            loss_list = [L1_Charbonnier_loss(sr, hr.to(device)) for sr, hr in zip(sr_list, hr_list) ]
            loss = sum(loss_list)            

            loss_count += loss.item()

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
    losslogger['epoch'].append(epoch)
    losslogger['loss'].append(loss_count/len(train_loader))       

#-------------------------------------------------------------------------#
print('==> training')

for epoch in range(opt.resume_epoch+1, 700): 
 
    train(epoch)
    
#     checkpoint
    if epoch % opt.num_cp == 0:        
        model_save_path = join(model_dir,"model_epoch_{}.pth".format(epoch))        
        state = {'epoch':epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'losslogger': losslogger,}
        torch.save(state,model_save_path)
        print("checkpoint saved to {}".format(model_save_path))     

    if epoch % opt.num_snapshot == 0:   
        plt.figure()
        plt.title('loss')
        plt.plot(losslogger['epoch'],losslogger['loss'])
        plt.savefig(model_dir+".jpg")
        plt.close()
        

