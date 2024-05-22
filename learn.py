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

HW = 224
# 数据增广
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomRotation(50),
    transforms.ToTensor()
])  # 这是一个流水线，注意是[]而不是{}，找了半天的bug

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


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
            print("共读入了%d张照片" % len(Y))
            return X, Y

    def __getitem__(self, item):
        if self.mode == "semi":
            return self.transform(self.X[item]), self.X[item]
        else:
            return self.transform(self.X[item]), self.Y[item]

    def __len__(self):
        return len(self.X)

class semiDataset(Dataset):
    def __init__(self, no_label_loader, model, device, thres=0.99):
        X, Y = self.data_pred(no_label_loader, model, device, thres)
        if X == []:
            self.flag = False
        else:
            self.flag = True
            self.X = np.array(X)
            self.Y = torch.LongTensor(Y)
            self.transform = train_transform

    def data_pred(self, no_label_loader, model, device, thres):    # 给无标签数据打标签
        model = model.to(device)
        soft = nn.Softmax(dim=1)
        pred_prob = []
        labels = []
        x = []
        y = []
        with torch.no_grad():
            for data in no_label_loader:
                data = data[0].to(device)
                pred = model(data)
                pred_soft = soft(pred)
                pred_max, pred_value = pred_soft.max(1)  # max函数会返回最大值和最大值的下标
                pred_prob.extend(pred_max.cpu().numpy().tolist())  # pred_prob保存的是所有样本的最大的概率
                labels.extend(pred_value.cpu().numpy().tolist())  # labels保存的是所有样本的预测标签
        for index, prob in enumerate(pred_prob):
            if prob > thres:
                x.append(no_label_loader.dataset[index][1])  # x是超过可信度的样本
                y.append(labels[index])  # y是超过可信度的样本的标签
        return x, y

    def __getitem__(self, item):
        return self.transform(self.X[item]), self.Y[item]

    def __len__(self):
        return len(self.Y)

def get_semi_loader(no_label_loader, model, device, thres):
    semi_data = semiDataset(no_label_loader, model, device, thres)
    if semi_data.flag == False:
        return None
    else:
        semi_loader = DataLoader(semi_data, batch_size=16, shuffle=False)
        return semi_loader



class myModel(nn.Module):
    def __init__(self, num_class):
        super(myModel, self).__init__()
        # 图片大小是3*224*224,输出维度计算公式 (i-k+2p)/s+1

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # (224-3+2)/1+1=224
            nn.BatchNorm2d(64),  # 批量归一化
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 64*122*122

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 128*56*56

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 256*28*28

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 512*14*14

        self.pool5 = nn.MaxPool2d(2)  # 512*7*7=25088

        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool5(x)
        x = x.view(x.size()[0], -1)  # 把矩阵拉直
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# 训练和验证函数
def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path, no_label_loader, thres):
    model = model.to(device)

    plt_train_loss = []
    plt_val_loss = []
    plt_train_acc = []
    plt_val_acc = []
    plt_semi_acc = []
    min_val_loss = 999999999
    max_val_acc = 0.0
    semi_loader = None

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        semi_loss = 0.0
        semi_acc = 0.0
        start_time = time.time()



        # 开始训练
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()  # 梯度清0
            x, y = batch_x.to(device), batch_y.to(device)
            pred = model(x)
            train_batch_loss = loss(pred, y)
            train_batch_loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数w,b
            train_loss += train_batch_loss.cpu().item()  # train_batch_loss表示一个batch的loss的和。train_loss是这轮训练中，所有batch的loss的和
            train_acc += np.sum(
                np.argmax(pred.detach().cpu().numpy(), axis=1) == batch_y.cpu().numpy())  # argmax用于找到最大值下标，
        plt_train_loss.append(train_loss / train_loader.dataset.__len__())  # 计算的是这一轮训练的平均loss
        plt_train_acc.append(train_acc / train_loader.dataset.__len__())

        if semi_loader != None:
            for batch_x, batch_y in semi_loader:
                optimizer.zero_grad()  # 梯度清0
                x, y = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                semi_batch_loss = loss(pred, y)
                semi_batch_loss.backward()  # 计算梯度
                optimizer.step()  # 更新参数w,b
                semi_loss += semi_batch_loss.cpu().item()
                semi_acc += np.sum(
                    np.argmax(pred.detach().cpu().numpy(), axis=1) == batch_y.cpu().numpy())
            plt_semi_acc.append(semi_acc / semi_loader.dataset.__len__())
            # print("半监督训练的准确率为：%.6f" % plt_semi_acc[-1])

        # 开始验证
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, y = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_batch_loss = loss(pred, y)
                val_loss += val_batch_loss.cpu().item()  # val_loss表示的是这个模型在这轮训练中，整个验证集上的loss的和
                val_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == batch_y.cpu().numpy())
        plt_val_loss.append(val_loss / val_loader.dataset.__len__())
        plt_val_acc.append(val_acc / val_loader.dataset.__len__())
        if val_acc > max_val_acc:
            torch.save(model, save_path)
            max_val_acc = val_acc
            semi_loader = None

        #  一轮结束后，输出这一轮的结果
        print("[%03d/%03d] %2.2f secs TrainLoss: %.6f ValLoss: %.6f TrainAcc: %.6f ValAcc: %.6f" % \
              (epoch + 1, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1], plt_train_acc[-1],
               plt_val_acc[-1]))

        # if plt_val_acc[-1] > 0.6:
        #     semi_loader = get_semi_loader(no_label_loader, model, device, thres)


    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss")
    plt.legend("train", "val")
    plt.show()

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title("acc")
    plt.legend("train", "val")
    plt.show()


config = {
    "lr": 0.001,
    "epochs": 20,
    "save_path": "model_save/best_model",
    "thres": 0.999,    # 实际中应该是0.9
    "momentum": 0.8
}

# 迁移学习
from model_utils.model import initialize_model
# model, _ = initialize_model("alexnet", 11, use_pretrained=True)
# model, _ = initialize_model("vgg", 11, use_pretrained=True)
model, _ = initialize_model("resnet18", 11, use_pretrained=True)

# model = myModel(11)
device = "cuda" if torch.cuda.is_available() else "cpu"


loss = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.AdamW(model.parameters(), config["lr"], weight_decay=0.0001)
# optimizer = optim.Adam(model.parameters(), config["lr"])
# optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

train_data = foodDataset(r"D:\pycharm\project\food\food-11\training\labeled", "train")
val_data = foodDataset(r"D:\pycharm\project\food\food-11\validation", "val")
no_label_data = foodDataset(r"D:\pycharm\project\food\food-11\training\unlabeled\00", "semi")

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True)
no_label_loader = DataLoader(no_label_data, batch_size=16, shuffle=False)


train_val(model, train_loader, val_loader, device, config["epochs"], optimizer, loss, config["save_path"], no_label_loader, config["thres"])
