'''依据原始论文搭建VGG网络'''
import torch.nn as nn
import torch
class VGG_16(nn.Module):
    def __init__(self, num_channels=3):  #由于传入的是RGB 所以传入通道为3
        super(VGG_16, self).__init__()
        #【input 244x244】 ->【conv（3）-（64） ，con（3）-（ 64）】->【maxpool】 3层 特征维度3->64
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=(3,3), padding='same')
        self.conv2 = nn.Conv2d(64, 64, (3,3), padding='same')
        #->【conv（3）-（128） ，con（3）-（128）】->【maxpool】 2层 特征维度64->128
        self.conv3 = nn.Conv2d(64, 128, (3,3), padding='same')
        self.conv4 = nn.Conv2d(128, 128, (3,3), padding='same')
        # ->【conv（3）-（256） ，con（3）-（256），conv（3）-（256）】->【maxpool】 2层 特征维度128->256
        self.conv5 = nn.Conv2d(128,256,(3,3),padding='same')
        self.conv6 = nn.Conv2d(256,256,(3,3),padding='same')
        self.conv7 = nn.Conv2d(256,256,(3,3),padding='same')
        #->【conv（3）-（512） ，con（3）-（512），conv（3）-（512）】->【maxpool】 2层 特征维度256->512
        self.conv8 = nn.Conv2d(256,512,(3,3),padding='same')
        self.conv9 = nn.Conv2d(512,512,(3,3),padding='same')
        self.conv10 = nn.Conv2d(512,512,(3,3),padding='same')
        #->【conv（3）-（512） ，con（3）-（512），conv（3）-（512）】->【maxpool】 2层 特征维度512->512
        self.conv11 = nn.Conv2d(512,512,(3,3),padding='same')
        self.conv12 = nn.Conv2d(512,512,(3,3),padding='same')
        self.conv13 = nn.Conv2d(512,512,(3,3),padding='same')
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.linear1 = nn.Linear(in_features=512 * 7 * 7, out_features=4096)
        self.linear2 = nn.Linear(in_features = 4096,out_features=4096)
        self.dropout = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(in_features=4096,out_features=1000)
        self.sofmax = nn.Softmax(dim = 0)

    def forward(self,input):  # input 为 RGB 图片，此时我们关注一下图片的大小
        temp  = self.relu(self.conv1(input)) #244x244
        temp  = self.relu(self.conv2(temp))
        temp  = self.maxpool(temp) # (M-K+2p)/S+1  （244-2）/2+1 =122  此时图片大小 122x122
        temp  = self.relu(self.conv3(temp))
        temp  = self.relu(self.conv4(temp))
        temp  = self.maxpool(temp)#  （122-2）/2+1 = 61  61x61
        temp  = self.relu(self.conv5(temp))
        temp  = self.relu(self.conv6(temp))
        temp  = self.relu(self.conv7(temp))
        temp  = self.maxpool(temp)#  向下取整（61-2）/2+1 =  30x30
        temp  = self.relu(self.conv8(temp))
        temp  = self.relu(self.conv9(temp))
        temp  = self.relu(self.conv10(temp))
        temp  = self.maxpool(temp) # （30-2）/2 +1 = 15  15x15
        temp  = self.relu(self.conv11(temp))
        temp  = self.relu(self.conv12(temp))
        temp  = self.relu(self.conv13(temp))
        temp  = self.maxpool(temp)# （15-2）/2+1 = 7x7
        # torch.Size([1, 512, 7, 7])
        temp = temp.reshape(-1)  # 一维  25088
        temp = self.relu(self.linear1(temp))
        temp = self.dropout(temp)
        temp = self.relu(self.linear2(temp))
        temp = self.dropout(temp)
        temp = self.linear3(temp)
        temp = self.sofmax(temp)
        return temp

'''生成一个VGG16 需要的图片数据'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('Pyramid.jpg') #此图片为 H x W x C 600x370x3 ，读入通道为BGR
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
'''图像金字塔,高斯金字塔,按比例上升，缩放'''
# img_down = cv2.pyrDown()
# print(img_down.shape) # 300,185,3
'''图像指定大小 resize 函数  cv2.resize(image,None,fx=int or float,fy=int or float)'''
re_img = cv2.resize(img,(244,244))
re_img1 = cv2.cvtColor(re_img,cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.title('row_img')
plt.imshow(img1)
plt.subplot(1,2,2)
plt.title('re_img')
plt.imshow(re_img1)
plt.show()
'''数据格式处理'''
IMG  =  np.zeros((1,3,244,244))
R,G,B = re_img1[:,:,0],re_img1[:, :,1 ],re_img1[:, :,2 ]
IMG[:,0,:,:],IMG[:,1,:,:],IMG[:,2,:,:] = R,G,B
IMG = torch.tensor(IMG)
IMG = IMG.to(torch.float32)
IMG.shape,type(IMG[0,0,0,0])
'''测试VGG16网络'''
vgg16 = VGG_16(num_channels=3)
out = vgg16.forward(input=IMG)
print(out.shape) #