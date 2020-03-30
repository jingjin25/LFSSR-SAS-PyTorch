
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1 
    else:
        center = factor - 0.5
        
    og = np.ogrid[:size, :size]
    filter = ( 1 - abs(og[0] - center) / factor ) * \
             ( 1 - abs(og[1] - center) / factor )
             
    return torch.from_numpy(filter).float()

class AltFilter(nn.Module):
    def __init__(self, an):
        super(AltFilter, self).__init__()
        
        self.an = an
        self.relu = nn.ReLU(inplace=True)
        self.spaconv = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.angconv = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):

        N, c, h, w = x.shape #[N*an2,c,h,w]
        N = N // (self.an*self.an)
        
        out = self.relu(self.spaconv(x)) #[N*an2,c,h,w]
        out = out.view(N, self.an*self.an, c, h*w)
        out = torch.transpose(out, 1, 3)
        out = out.view(N*h*w, c, self.an, self.an)  #[N*h*w,c,an,an]

        out = self.relu(self.angconv(out)) #[N*h*w,c,an,an]
        out = out.view(N, h*w, c, self.an*self.an)
        out = torch.transpose(out, 1, 3)
        out = out.view(N*self.an*self.an, c, h, w) #[N*an2,c,h,w]
        return out


class net2x(nn.Module):
    
    def __init__(self, an, layer):        
        
        super(net2x, self).__init__()
        
        self.an = an 
        self.an2 = an * an
        self.relu = nn.ReLU(inplace=True)
        
        self.conv0 = nn.Conv2d(in_channels = 1,out_channels = 64,kernel_size=3,stride=1,padding=1)
        
        self.altblock1 = self.make_layer(layer_num=layer)
        self.fup1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64,out_channels = 64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            )
        self.res1 =  nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size = 3,stride = 1,padding = 1)    
        self.iup1 =  nn.ConvTranspose2d(in_channels = 1,out_channels = 1,kernel_size=4,stride=2,padding=1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
        
    def make_layer(self, layer_num):
        layers = []
        for i in range( layer_num ):
            layers.append( AltFilter( self.an ) )
        return nn.Sequential(*layers)     
            
    def forward(self, lr):
    
        N,_,h,w = lr.shape   #lr [N,81,h,w]  
        lr = lr.view(N*self.an2,1,h,w)  #[N*81,1,h,w]                          

        x = self.relu(self.conv0(lr)) #[N*81,64,h,w]
        f_1 = self.altblock1(x) #[N*81,64,h,w]
        fup_1 = self.fup1(f_1)  #[N*81,64,2h,2w]
        res_1 = self.res1(fup_1)  #[N*81,1,2h,2w]
        iup_1 = self.iup1(lr)   #[N*81,1,2h,2w]
        
        sr_2x = res_1 + iup_1   #[N*81,1,2h,2w]
        sr_2x = sr_2x.view(N,self.an2,h*2,w*2)
        return [sr_2x]      


class net4x(nn.Module):
    
    def __init__(self, an, layer):        
        
        super(net4x, self).__init__()
        
        self.an = an 
        self.an2 = an * an
        self.relu = nn.ReLU(inplace=True)
        
        self.conv0 = nn.Conv2d(in_channels = 1,out_channels = 64,kernel_size=3,stride=1,padding=1)
        
        self.altblock1 = self.make_layer(layer_num=layer)
        self.fup1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64,out_channels = 64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            )
        self.res1 =  nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size = 3,stride = 1,padding = 1)    
        self.iup1 =  nn.ConvTranspose2d(in_channels = 1,out_channels = 1,kernel_size=4,stride=2,padding=1)
        
        self.altblock2 = self.make_layer(layer_num=layer)
        self.fup2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64,out_channels = 64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            )
        self.res2 =  nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size = 3,stride = 1,padding = 1)    
        self.iup2 =  nn.ConvTranspose2d(in_channels = 1,out_channels = 1,kernel_size=4,stride=2,padding=1)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def make_layer(self, layer_num):
        layers = []
        for i in range( layer_num ):
            layers.append( AltFilter( self.an ) )
        return nn.Sequential(*layers)     
            
    def forward(self, lr):
    
        N,_,h,w = lr.shape   #lr [N,81,h,w]  
        lr = lr.view(N*self.an2,1,h,w)  #[N*81,1,h,w]
        
        x = self.relu(self.conv0(lr)) #[N*81,64,h,w]
        f_1 = self.altblock1(x) #[N*81,64,h,w]
        fup_1 = self.fup1(f_1)  #[N*81,64,2h,2w]
        res_1 = self.res1(fup_1)  #[N*81,1,2h,2w]
        iup_1 = self.iup1(lr)   #[N*81,1,2h,2w]

        sr_2x = res_1 + iup_1   #[N*81,1,2h,2w]
         
        f_2 = self.altblock2(fup_1)  #[N*81,64,2h,2w]
        fup_2 = self.fup2(f_2)     #[N*81,64,4h,4w]
        res_2 = self.res2(fup_2)  #[N*81,1,4h,4w]
        iup_2 = self.iup2(sr_2x)  #[N*81,1,4h,4w]
        sr_4x = res_2 + iup_2   #[N*81,1,4h,4w]

        sr_2x = sr_2x.view(N,self.an2,h*2,w*2)
        sr_4x = sr_4x.view(N,self.an2,h*4,w*4)   
        
        return sr_4x, sr_2x