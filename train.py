import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from efficientnet_pytorch import EfficientNet
import time


class BirdDataset(Dataset):

    def __init__(self, root, transform):

        self.transform = transform  # 影像的轉換方式

        fh = open('2021VRDL_HW1_datasets/classes.txt')
        d = {}
        idx = 0
        for line in fh.readlines():
            cls = line.split()
            d[cls.pop(0)] = idx
            idx = idx + 1

        fh = open(root)
        labels = []
        imgs = []
        for line in fh.readlines():
            cls = line.split()
            # image name
            imgs.append('2021VRDL_HW1_datasets/training_images/'+cls.pop(0))
            label = d[cls.pop(0)]
            labels.append(label)  # image label

        self.filenames = imgs  # 資料集的所有檔名
        self.labels = labels  # 影像的標籤
        assert len(self.filenames) == len(self.labels), 'mismatched length!'

    def __len__(self):

        return len(self.filenames)  # return DataSet 長度

    def __getitem__(self, idx):

        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)  # Transform image
        label = np.array(self.labels[idx])

        return image, label  # return 模型訓練所需的資訊


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Create the dic: name of class to index
# ex : 001.Black_footed_Albatross to 0
#      ...                        to .
#      200.Common_Yellowthroat    to 199

fh = open('2021VRDL_HW1_datasets/classes.txt')
dic = {}
idx = 0
for line in fh.readlines():
    cls = line.split()
    dic[cls.pop(0)] = idx
    idx += 1
Counter = [0] * 200

# the data in training_labels.txt has 15 imgs for each class (uniformly)
# there are 200 classes
# Spilt the training_labels.txt to train and val
# train is 14 x 200
# val is 1 x 200

fh = open('2021VRDL_HW1_datasets/training_labels.txt')
train = []
val = []
for line in fh.readlines():

    cls = line.split()
    image_name = cls.pop(0)  # image name
    # print(cls.pop(0))
    label = cls.pop(0)
    label_id = dic[label]
    Counter[label_id] = Counter[label_id] + 1
    # there are 15 for each class
    if(Counter[label_id] == 5):
        val.append([image_name, label])
    else:
        train.append([image_name, label])

np.savetxt('2021VRDL_HW1_datasets/train_data.txt', train, fmt='%s')
np.savetxt('2021VRDL_HW1_datasets/val_data.txt', val, fmt='%s')

batch_size = 8
lr = 1e-3
epochs = 1
image_size = 456

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data_transformer = transforms.Compose([
    transforms.Resize((image_size)),
    transforms.RandomResizedCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.RandomErasing(inplace=True),
    normalize
])
val_transformer = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    normalize
])
test_transformer = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    normalize
])
# 14 * 20 = 2800
# create dataset
training_set = BirdDataset(
    root='2021VRDL_HW1_datasets/train_data.txt', transform=data_transformer)
train_dataloader = torch.utils.data.DataLoader(
    training_set, batch_size=batch_size, shuffle=True, pin_memory=True)

# 1 * 20 = 200
# create dataset
val_set = BirdDataset(
    root='2021VRDL_HW1_datasets/val_data.txt', transform=val_transformer)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size*2, shuffle=True, pin_memory=True)

C = EfficientNet.from_pretrained('efficientnet-b5', num_classes=200)
optimizer_C = optim.SGD(
    C.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
criteron = nn.CrossEntropyLoss()
lr_scheduler = optim.lr_scheduler.ExponentialLR(
    optimizer_C, gamma=0.1, last_epoch=-1)
C = C.cuda()

loss_epoch_C = []
train_acc, test_acc = [], []
best_acc, best_auc = 0.0, 0.0

if __name__ == '__main__':

    for epoch in range(epochs):

        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss_C = 0.0

        C.train()  # 設定 train 或 eval

        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))

        # ---------------------------
        # Training Stage
        # ---------------------------

        start = time.time()
        for i, (x, label) in enumerate(train_dataloader):

            x, label = x.to(device), label.to(device)
            optimizer_C.zero_grad()  # 清空梯度

            output = C(x)  # 將訓練資料輸入至模型進行訓練
            loss = criteron(output, label)  # 計算 loss

            loss.backward()  # 將 loss 反向傳播
            optimizer_C.step()  # 更新權重
            # lr_scheduler.step()

            # 計算訓練資料的準確度 (correct_train / total_train)
            _, predicted = torch.max(output.data, 1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum()
            train_loss_C += loss.item()
            iter += 1
        end = time.time()
        print('Training epoch: %d / loss_C: %.3f | acc: %.3f | time:%.3f' %
              (epoch + 1, train_loss_C / iter,
               correct_train / total_train, end-start))

        #  --------------------------
        #  Testing Stage
        #  --------------------------

        C.eval()  # 設定 train 或 eval

        start = time.time()
        for i, (x, label) in enumerate(val_dataloader):

            with torch.no_grad():  # 測試階段不需要求梯度
                x, label = x.to(device), label.to(device)

                output = C(x)  # 將測試資料輸入至模型進行測試
                # 計算測試資料的準確度
                _, predicted = torch.max(output.data, 1)
                total_test += label.size(0)
                correct_test += (predicted == label).sum()
        end = time.time()
        print('Testing acc: %.3f | time:%.3f' % (
            correct_test / total_test, end-start))

        train_acc.append(100 * (correct_train / total_train))
        test_acc.append(100 * (correct_test / total_test))
        loss_epoch_C.append(train_loss_C / iter)  # loss

    torch.save(C, 'model_save/'+str(epoch)+'.pth')

    plt.figure()
    plt.plot(loss_epoch_C)  # plot your loss

    plt.title('Training Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss_C'], loc='upper left')
    plt.savefig('Loss.png')
    plt.show()
    plt.figure()

    plt.plot(train_acc, label="train")  # plot your training accuracy
    plt.plot(test_acc, label="test")  # plot your testing accuracy
    plt.plot([0, epochs], [70, 70], 'k-', lw=1, dashes=[2, 2])

    plt.title('Training acc')
    plt.ylabel('acc (%)'), plt.xlabel('epoch')
    plt.legend(['training acc', 'testing acc'], loc='upper left')
    plt.savefig('Acc.png')
    plt.show()
