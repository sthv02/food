import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import transforms
from torch import optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt

def seed_everything(seed):  # 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(0)  # 让代码每次运行结果一样，


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(50),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

HW = 224


class foodDataset(Dataset):
    def __init__(self, file_path, mode):
        self.mode = mode
        if mode == "train":
            self.transform = train_transform
        else:
            self.transform = val_transform
        if mode == "semi":
            self.X = self.read_file(file_path)
        else:
            x, y = self.read_file(file_path)
            self.X = x
            self.Y = torch.LongTensor(y)

    def read_file(self, path):
        if self.mode == "semi":
            file_list = os.listdir(path)
            file_len = len(file_list)
            X = np.zeros((file_len, HW, HW, 3), dtype=np.uint8)
            for j, each in enumerate(file_list):  # enumerate是用来返回索引对应的下标和值
                img_path = path + "/" + each
                img = Image.open(img_path)
                img = img.resize((HW, HW))  # 图像识别一般都固定图像大小为224*224
                X[j, ...] = img
            return X
        else:
            for i in tqdm(range(11)):
                file_path = path + "/%02d/" % i  # i代表第几类的图片
                file_list = os.listdir(file_path)  # 读取文件夹下所有文件
                file_len = len(file_list)
                xi = np.zeros((file_len, HW, HW, 3), dtype=np.uint8)
                yi = np.zeros(file_len, dtype=np.uint8)
                for j, each in enumerate(file_list):  # enumerate是用来返回索引对应的下标和值
                    img_path = file_path + "/" + each
                    img = Image.open(img_path)
                    img = img.resize((HW, HW))  # 图像识别一般都固定图像大小为224*224
                    xi[j, ...] = img
                    yi[j] = i
                if i == 0:
                    X = xi
                    Y = yi
                else:
                    X = np.concatenate((X, xi), axis=0)
                    Y = np.concatenate((Y, yi), axis=0)
            return X, Y

    def __getitem__(self, item):
        if self.mode == "semi":
            return self.transform(self.X[item]), self.X[item]
        else:
            return self.transform(self.X[item]), self.Y[item]

    def __len__(self):
        return len(self.Y)
