import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os,argparse
import numpy as np
from PIL import Image
from models.DustFormer import DustFormer

####get your current work path
abs= os.getcwd() + '/'

def tensorShow(tensors,titles=['haze']):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

####task setting, please change the path of your test images path 
parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='dust',help='dust or dense or nhhaze ')
parser.add_argument('--test_imgs',type=str,default='datasets/hazy',help='Test imgs folder')
opt=parser.parse_args()
dataset=opt.task  

img_dir = abs + opt.test_imgs +'/'  

####please change the dedust results output_dir
output_dir = abs + f'dedustresult/'
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

####please change your trained_model path 
model_dir='./trained_models/'+f'dustformer.pk'
### we have train our model on two datasets: RB-Dust dataset and SID dataset, 
### please see the link in ReadMe file to find the google drive link to download them
### link is here https://drive.google.com/drive/folders/1g6X7wYV_yILRqYx7GoUuhW4CSJXiPnCZ?usp=sharing

device='cuda' if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir,map_location=device)
net=DustFormer()
net=nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()

for im in os.listdir(img_dir):
    print(f'\r {im}',end='',flush=True)
    haze = Image.open(img_dir+im)
    haze1= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None,::]
    haze_no=tfs.ToTensor()(haze)[None,::]
    with torch.no_grad():
        pred = net(haze1)
    ts=torch.squeeze(pred.clamp(0,1).cpu())
    vutils.save_image(ts,output_dir+im.split('.')[0]+'.png')
   
####Please go to the “dedustresult” folder for test results.
