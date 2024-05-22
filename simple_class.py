import torch
from torch.utils.data import Dataset,DataLoader
import random
import numpy as np
import os
from PIL import Image
from tqdm import tqdm   # 看进度条的包
from torchvision.transforms import transforms
import torch.nn as nn
import time
import matplotlib.pyplot as plt

train_transformer = transforms.Compose([
    transforms.ToPILImage(),  # 这个img 转为PIL(224*224*3 转为 3*224*224)
    transforms.RandomResizedCrop(224),  #随机resize 。 取中间的224.
    transforms.ToTensor()          # 转为张量

])


def seed_everything(seed):      # 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
#################################################################
seed_everything(0)   # 让代码每次运行结果一样，
########################

#######读数据。
HW = 224
def read_file(path):
    for i in tqdm(range(11)):
        img_path = path + "/%02d/"%i
        img_list = os.listdir(img_path) # 读取文件夹下所有文件
        xi = np.zeros((len(img_list), HW, HW, 3), dtype=np.uint8)  #280 * 224* 224 *3
        yi = np.zeros(len(img_list), dtype=np.uint8)# 280*1
        for j, img in enumerate(img_list):
            true_img_path = img_path + img
            img = Image.open(true_img_path)
            img = img.resize((HW,HW))       # 转为224*224
            xi[j, ...] = img
            yi[j] = i

        if i == 0:
            X = xi
            Y = yi
        else:
            X = np.concatenate((X, xi), axis=0)
            Y = np.concatenate((Y, yi), axis=0)
    print("读取了%d个数据"%len(Y))
    return  X, Y

# read_file(r"F:\pycharm\beike\classification\food_classification\food-11\training\labeled")
#D:\pycharm\project\food\food-11\training\labeled

class fooddataset(Dataset):
    def __init__(self, path):
        super(fooddataset, self).__init__()
        self.X, self.Y = read_file(path)
        self.Y = torch.LongTensor(self.Y)
        self.transformer = train_transformer

    def __getitem__(self, item):
        return self.transformer(self.X[item]), self.Y[item]

    def __len__(self):
        return len(self.Y)


train_set = fooddataset(r"F:\pycharm\beike\classification\food_classification\food-11_sample\training\labeled")
val_set = fooddataset(r"F:\pycharm\beike\classification\food_classification\food-11_sample\validation")

train_loader = DataLoader(train_set, batch_size=4)
val_loader = DataLoader(val_set, batch_size=4)


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        #3*224*224 - 512*7*7
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),      # 输入维度 输出维度（channel） 卷积核大小 padding stride
            nn.BatchNorm2d(64),     # 特征归一化
            nn.ReLU(),
            nn.MaxPool2d(2)     # 最大池化，把一个2*2变成1*1
        )  # 把里面的操作合成一个    64*112*112

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 把里面的操作合成一个    128*56*56

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 把里面的操作合成一个    256*28*28

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 把里面的操作合成一个    512*14*14
        self.pool1 = nn.MaxPool2d(2) #512*7*7
        self.fc1 = nn.Linear(25088, 1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 11)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool1(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

mymodel = myModel()
#
# for data in train_loader:
#     print(data)
#     x, y = data[0], data[1]
#     out = mymodel(x)


lr = 0.001
loss = nn.CrossEntropyLoss()
epochs = 20
optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr, weight_decay=1e-4)             # AdamW
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_ = 'model_save/mymodel.pth'



def train_val(model, trainloader, valloader,optimizer, loss, epoch, device, save_):
    # trainloader = DataLoader(trainset,batch_size=batch,shuffle=True)
    # valloader = DataLoader(valset,batch_size=batch,shuffle=True)
    model = model.to(device)                # 模型和数据 ，要在一个设备上。  cpu - gpu
    plt_train_loss = []
    plt_val_loss = []


    val_rel = []
    min_val_loss = 100000                 # 记录训练验证loss 以及验证loss和结果

    for i in range(epoch):                 # 训练epoch 轮
        train_acc = 0.0
        val_acc = 0.0
        start_time = time.time()             # 记录开始时间
        model.train()                         # 模型设置为训练状态      结构
        train_loss = 0.0
        val_loss = 0.0
        for data in trainloader:                     # 从训练集取一个batch的数据
            optimizer.zero_grad()                   # 梯度清0
            x, target = data[0].to(device), data[1].to(device)       # 将数据放到设备上
            pred = model(x)                          # 用模型预测数据
            bat_loss = loss(pred, target)       # 计算loss
            bat_loss.backward()                        # 梯度回传， 反向传播。
            optimizer.step()                            #用优化器更新模型。  轮到SGD出手了
            train_loss += bat_loss.detach().cpu().item()             #记录loss和
            train_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == data[1].cpu().numpy())

        plt_train_loss. append(train_loss/trainloader.dataset.__len__())   #记录loss到列表。注意是平均的loss ，因此要除以数据集长度。

        model.eval()                 # 模型设置为验证状态
        with torch.no_grad():                    # 模型不再计算梯度
            for data in valloader:                      # 从验证集取一个batch的数据
                val_x , val_target = data[0].to(device), data[1].to(device)          # 将数据放到设备上
                val_pred = model(val_x)                 # 用模型预测数据
                val_bat_loss = loss(val_pred, val_target)          # 计算loss
                val_loss += val_bat_loss.detach().cpu().item()                  # 计算loss
                val_rel.append(val_pred)                 #记录预测结果
        if val_loss < min_val_loss:
            torch.save(model, save_)               #如果loss比之前的最小值小， 说明模型更优， 保存这个模型
        val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == val_target[1].cpu().numpy())
        print("val_acc：",val_acc)
        plt_val_loss.append(val_loss/valloader.dataset.__len__())  #记录loss到列表。注意是平均的loss ，因此要除以数据集长度。
        #
        print('[%03d/%03d] %2.2f sec(s) TrainLoss : %.6f | valLoss: %.6f' % \
              (i, epoch, time.time()-start_time, plt_train_loss[-1], plt_val_loss[-1])
              )              #打印训练结果。 注意python语法， %2.2f 表示小数位为2的浮点数， 后面可以对应。

        # print('[%03d/%03d] %2.2f sec(s) TrainLoss : %3.6f | valLoss: %.6f' % \
        #       (i, epoch, time.time()-start_time, 2210.2255411, plt_val_loss[-1])
        #       )              #打印训练结果。 注意python语法， %2.2f 表示小数位为2的浮点数， 后面可以对应。
    plt.plot(plt_train_loss)              # 画图， 向图中放入训练loss数据
    plt.plot(plt_val_loss)                # 画图， 向图中放入训练loss数据
    plt.title('loss')                      # 画图， 标题
    plt.legend(['train', 'val'])             # 画图， 图例
    plt.show()                                 # 画图， 展示


train_val(mymodel, train_loader, val_loader,optimizer, loss, epochs, device, save_)