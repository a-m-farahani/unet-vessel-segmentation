import numpy as np
import PIL
from PIL import Image
from PIL import ImageStat
import os
import random
import torch
import torchvision

class DRIVE:
    def __init__(self,dataset_path,resize_to=None):
        self.train_images_path = dataset_path + "/training/images/"
        self.train_seg_path = dataset_path + "/training/1st_manual/"
        self.test_images_path = dataset_path + "/test/images/"
        self.test_seg_path = dataset_path + "/test/1st_manual/"

        self.Train = ImageLoader(self.train_images_path,self.train_seg_path,resize_to)
        self.Test = ImageLoader(self.test_images_path,self.test_seg_path,resize_to)
    def reset(self):
      self.Train.reset()
      self.Test.reset()
      

class ImageLoader:
    def __init__(self,images_path,seg_path,d_size=None):
        self.cursor = 0
        self.img_path = images_path
        self.seg_path = seg_path
        self.n = len(os.listdir(images_path))
        self.d_size = d_size
        self.img_list = os.listdir(self.img_path)
        random.shuffle(self.img_list)
        self.batch_size = 1

    def reset(self):
      self.cursor = 0
      random.shuffle(self.img_list)

    def next_image(self):
        if self.cursor >= self.n:
            self.cursor = 0
        ss = self.img_list[self.cursor]
        ss = ss[0:2] + "_manual1.gif"
        x = Image.open(self.img_path+self.img_list[self.cursor],mode='r')
        y = Image.open(self.seg_path+ss,mode='r').convert('L')
        self.cursor += 1

        if self.d_size is not None:
            x = x.resize((self.d_size[0],self.d_size[1]),Image.NEAREST)
            y = y.resize((self.d_size[0],self.d_size[1]),Image.NEAREST)

        return x,y,ss
      
    def get_mean(self):
        img,_,_ = self.next_image()
        mean = ImageStat.Stat(img).mean
        for _ in range(self.n):
            img,_,_ = self.next_image()
            st = ImageStat.Stat(img)
            for j in range(len(mean)):
                mean[j] += st.mean[j]
        for j in range(len(mean)):
            mean[j] /= self.n
        return list(map(lambda x:x/255,mean))

    def get_std(self):
        img,_,_ = self.next_image()
        std = ImageStat.Stat(img).stddev
        for _ in range(self.n):
            img,_,_ = self.next_image()
            st = ImageStat.Stat(img)
            for j in range(len(std)):
                std[j] += st.stddev[j]
        for j in range(len(std)):
            std[j] /= self.n
        return list(map(lambda x:x/255,std))
    
    def next_batch(self,batch_size):      
      self.batch_size = batch_size
      imgs = torch.zeros(batch_size,3,height,width)
      segs = torch.zeros(batch_size,1,height,width)
      for i in range(batch_size):
        img,seg,_ = self.next_image()
        imgs[i,:,:,:] = torchvision.transforms.to_tensor(img)
        segs[i,0,:,:] = torchvision.transforms.to_tensor(seg)
      return imgs,segs
