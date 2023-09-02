import torch.utils.data as data
import torchvision.transforms as tfs
import numpy as np
from torchvision.transforms import functional as FF
import os
import random
from PIL import Image
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

from torchvision.utils import make_grid
def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

class RESIDE_Dataset(data.Dataset):
    def __init__(self,path,train, size='whole img', format='.png'):
        super(RESIDE_Dataset,self).__init__()
        self.size=size
        #print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'hazy'))
        #print('self_haze_imgs_dir :', self.haze_imgs_dir)
        
        self.haze_imgs=[os.path.join(path,'hazy',img) for img in self.haze_imgs_dir]
        #print('self_haze_imgs:', self.haze_imgs) 
        
        self.clear_dir=os.path.join(path,'clear')
    def __getitem__(self, index):#调用__getitem__(index) 
       
        haze=Image.open(self.haze_imgs[index])
        if isinstance(self.size,int):# 如果size是int型，则返回True
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(0,20000)
                
                haze=Image.open(self.haze_imgs[index])
       
        img=self.haze_imgs[index]
        #id=img.split('/')[-1].split('.')[0]
        id=img.split('/')[-1].split('.')[0] 
        #middle= img.split('/')[-1].split('_')[1]
        #last = img.split('/')[-1].split('_')[-1]
        filename=img.split('/')[-1]
        if self.train:
            clear_name=id+self.format#self.format#
        else :
            clear_name=id+ self.format#
            #clear_name=id+ self.format
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return {'haze': haze,'clear': clear, 'filename': filename}
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot) 
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
 
      
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)

path='/home/yons/home/zslinshi/dedust/net/dataset'
def dataset_select(dataset, is_train, opt):
    if is_train:
        crop_size = opt.crop_size
        if dataset == 'nhhaze':
            train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/Realworld/nhhaze',train=True,size=crop_size),
                                         batch_size=opt.bs,shuffle=True)
        elif dataset == 'dense':
            train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/Realworld/dense',train=True,size=crop_size),
                                      batch_size=opt.bs,shuffle=True)
        elif dataset == 'dust':
            train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/Realworld/dust',train=True,size=crop_size),
                                      batch_size=opt.bs,shuffle=True)            
        else: train_loader = None
        return train_loader
    else:
        if dataset=='nhhaze':
            test_loader = DataLoader(
                dataset=RESIDE_Dataset(path + '/Realworld/nhhazetest', train=False, size='whole img', format='.png'),
                batch_size=1, shuffle=False)
        elif dataset=='dense':
          test_loader = DataLoader(
              dataset=RESIDE_Dataset(path + '/Realworld/densetest', train=False, size='whole img', format='.png'),
              batch_size=1, shuffle=False)   
          
        elif dataset=='dust':
          test_loader = DataLoader(
              dataset=RESIDE_Dataset(path + '/Realworld/dusttest', train=False, size='whole img', format='.png'),
              batch_size=1, shuffle=False)   
          