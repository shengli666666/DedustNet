from Dwt import DWTForward
import torch
import cv2
import torchvision


img = cv2.imread('./1.jpg')
img = cv2.cvtColor (img, cv2 .COLOR_BGR2RGB)
x = torch.from_numpy(img)
x = x/255.0
print(x.size())
x = x.permute(2, 0, 1)
x = x.unsqueeze(0)
print(x.size())

dwt1 = DWTForward(J=1, wave='db1', mode='zero')

yl, yh = dwt1(x)
print(yl.size())
b, c, h, w = yl.size()
for i in range(len(yh)):
    yh_re = yh[i].reshape(b, c * 3, h, w)
    print(yh[i].size())
    #print(yh_re.size())

    yhl = yh[0][:, :, 0, :, :]
    ylh = yh[0][:, :, 1, :, :]
    yhh = yh[0][:, :, 2, :, :]

y = torch.cat((yl, yh_re), 1)
print(y.size())

x = torchvision.transforms.Resize([h,w])(x)
torchvision.utils.save_image(torch.cat((x, yl, yhl, ylh, yhh), 0), "results/" + '1.png')

yl, yh = dwt1(y)
print(yl.size())
b, c, h, w = yl.size()
for i in range(len(yh)):
    yh_re = yh[i].permute(0, 2, 1, 3, 4).reshape(b, c * 3, h, w)
y = torch.cat((yl, yh_re), 1)
print(y.size())