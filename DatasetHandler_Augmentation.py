import numpy as np
from PIL import Image
import os
import torch
import torchvision
from torch.utils.data import Dataset
import time
import random

class DriveTrainDataset(Dataset):
  def __init__(self,root_path,transform=None):
    self.root = root_path + "/training"
    self.n = len(os.listdir(self.root+"/images"))
    self.samples = os.listdir(self.root+"/images")
    self.transform = transform
    if transform is None:
      self.transform = torchvision.transform.ToTensor()
  
  def __len__(self):
    return self.n
  
  def __getitem__(self,index):
    if index >= self.n:
      index = 0
    temp = self.samples[index]
    img = Image.open(self.root+"/images/"+temp,mode='r')
    temp = temp[0:2]+"_manual1.gif"
    seg = Image.open(self.root+"/1st_manual/"+temp,mode='r').convert('L')
    
    seed = time.time()
    random.seed(seed)
    img = self.transform(img)
    random.seed(seed)
    seg = self.transform(seg)
    return (img,seg)
  
class DriveTestDataset(Dataset):
  def __init__(self,root_path,transform=None):
    self.root = root_path + "/test"
    self.n = len(os.listdir(self.root+"/images"))
    self.samples = os.listdir(self.root+"/images")
    self.transform = transform
    if transform is None:
      self.transform = torchvision.transform.ToTensor()
  
  def __len__(self):
    return self.n
  
  def __getitem__(self,index):
    if index >= self.n:
      index = 0
    temp = self.samples[index]
    img = Image.open(self.root+"/images/"+temp,mode='r')
    temp = temp[0:2]+"_manual1.gif"
    seg = Image.open(self.root+"/1st_manual/"+temp,mode='r').convert('L')
    
    seed = time.time()
    random.seed(seed)
    img = self.transform(img)
    random.seed(seed)
    seg = self.transform(seg)
    return (img,seg)
  
class STARE:
  def __init__(self,root_path,train_test_ratio=0.75,transform=None):
    self.root = root_path
    self.transform = transform
    self.sample_list = os.listdir(self.root+"/images")
    self.n = len(self.sample_list)
    randperm = np.random.permutation(self.n)
    self.nTrain = int(self.n * train_test_ratio)
    self.nTest = self.n - self.nTrain
    self.Train = StareDataset(self.root,self.sample_list[0:self.nTrain],self.transform)
    self.Test = StareDataset(self.root,self.sample_list[self.nTrain:],self.transform)
    

class StareDataset(Dataset):
  def __init__(self,root_path,sample_list,transform=None):
    self.root = root_path
    self.samples = sample_list
    self.n = len(self.samples)
    self.transform = transform
    if transform is None:
      self.transform = torchvision.transform.ToTensor()
  
  def __len__(self):
    return self.n
  
  def __getitem__(self,index):
    if index >= self.n:
      index = 0
    temp = self.samples[index]
    img = Image.open(self.root+"/images/"+temp,mode='r')
    temp = temp[0:6]+".ah.ppm"
    seg = Image.open(self.root+"/labels/"+temp,mode='r').convert('L')
    
    seed = time.time()
    random.seed(seed)
    img = self.transform(img)
    random.seed(seed)
    seg = self.transform(seg)
    return (img,seg)
  
class CHASE:
  def __init__(self,root_path,train_test_ratio=0.75,transform=None):
    self.root = root_path
    self.transform = transform
    self.sample_list = os.listdir(self.root+"/images")
    self.n = len(self.sample_list)
    randperm = np.random.permutation(self.n)
    self.nTrain = int(self.n * train_test_ratio)
    self.nTest = self.n - self.nTrain
    self.Train = ChaseDataset(self.root,self.sample_list[0:self.nTrain],self.transform)
    self.Test = ChaseDataset(self.root,self.sample_list[self.nTrain:],self.transform)
    

class ChaseDataset(Dataset):
  def __init__(self,root_path,sample_list,transform=None):
    self.root = root_path
    self.samples = sample_list
    self.n = len(self.samples)
    self.transform = transform
    if transform is None:
      self.transform = torchvision.transform.ToTensor()
  
  def __len__(self):
    return self.n
  
  def __getitem__(self,index):
    if index >= self.n:
      index = 0
    temp = self.samples[index]
    img = Image.open(self.root+"/images/"+temp,mode='r')
    temp = temp[0:9]+"_1stHO.png"
    seg = Image.open(self.root+"/labels/"+temp,mode='r').convert('L')
    
    seed = time.time()
    random.seed(seed)
    img = self.transform(img)
    random.seed(seed)
    seg = self.transform(seg)
    return (img,seg)
  
class HRF:
  def __init__(self,root_path,train_test_ratio=0.75,transform=None):
    self.root = root_path
    self.transform = transform
    self.sample_list = os.listdir(self.root+"/images")
    self.n = len(self.sample_list)
    randperm = np.random.permutation(self.n)
    self.nTrain = int(self.n * train_test_ratio)
    self.nTest = self.n - self.nTrain
    self.Train = HRFDataset(self.root,self.sample_list[0:self.nTrain],self.transform)
    self.Test = HRFDataset(self.root,self.sample_list[self.nTrain:],self.transform)
    

class HRFDataset(Dataset):
  def __init__(self,root_path,sample_list,transform=None):
    self.root = root_path
    self.samples = sample_list
    self.n = len(self.samples)
    self.transform = transform
    if transform is None:
      self.transform = torchvision.transform.ToTensor()
  
  def __len__(self):
    return self.n
  
  def __getitem__(self,index):
    if index >= self.n:
      index = 0
    temp = self.samples[index]
    img = Image.open(self.root+"/images/"+temp,mode='r')
    temp = temp[:-3]+"tif"
    seg = Image.open(self.root+"/manual1/"+temp,mode='r').convert('L')
    
    seed = time.time()
    random.seed(seed)
    img = self.transform(img)
    random.seed(seed)
    seg = self.transform(seg)
    return (img,seg)
  

class ARIA:
  def __init__(self,root_path,train_test_ratio=0.75,transform=None):
    self.root = root_path
    self.transform = transform
    self.sample_list = os.listdir(self.root+"/images")
    self.n = len(self.sample_list)
    randperm = np.random.permutation(self.n)
    self.nTrain = int(self.n * train_test_ratio)
    self.nTest = self.n - self.nTrain
    self.Train = ARIADataset(self.root,self.sample_list[0:self.nTrain],self.transform)
    self.Test = ARIADataset(self.root,self.sample_list[self.nTrain:],self.transform)

class ARIADataset(Dataset):
  def __init__(self,root_path,sample_list,transform=None):
    self.root = root_path
    self.samples = sample_list
    self.n = len(self.samples)
    self.transform = transform
    if transform is None:
      self.transform = torchvision.transform.ToTensor()
  
  def __len__(self):
    return self.n
  
  def __getitem__(self,index):
    if index >= self.n:
      index = 0
    temp = self.samples[index]
    img = Image.open(self.root+"/images/"+temp,mode='r')
    temp = temp[:-4]+"_BDP.tif"
    seg = Image.open(self.root+"/labels/"+temp.replace(" ",""),mode='r').convert('L')
    
    seed = time.time()
    random.seed(seed)
    img = self.transform(img)
    random.seed(seed)
    seg = self.transform(seg)
    return (img,seg)
  
  
  
compose = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(180,resample=Image.NEAREST),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.Resize((512,512),interpolation=Image.NEAREST),
    torchvision.transforms.ToTensor()
])

"""compose = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
"""
  
#ds = DriveTrainDataset("gdrive/My Drive/Archive/DataSets/DRIVE",transform=compose)
aria_ds = ARIA("gdrive/My Drive/Archive/DataSets/ARIA",transform=compose)
ds = aria_ds.Train;

ds_loader = torch.utils.data.DataLoader(dataset=ds,batch_size=8,shuffle=True)
