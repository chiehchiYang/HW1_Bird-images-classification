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


class Bird_test_Dataset(Dataset):

    def __init__(self, root, transform, is_train=True):

        self.transform = transform

        fh = open('2021VRDL_HW1_datasets/classes.txt')
        d = {}
        idx = 0
        for line in fh.readlines():
            cls = line.split()
            d[cls.pop(0)] = idx
            idx = idx + 1

        if is_train is True:
            fh = open(root)
            labels = []
            imgs = []
            for line in fh.readlines():
                cls = line.split()
                imgs.append(
                    '2021VRDL_HW1_datasets/training_images/' + cls.pop(0))

                label = d[cls.pop(0)]
                labels.append(label)  # image label

            self.filenames = imgs  # 資料集的所有檔名
            self.labels = labels  # 影像的標籤
            assert len(self.filenames) == len(self.labels), 'mismatched'
        else:
            fh = open(root)
            labels = []
            imgs = []
            for line in fh.readlines():
                cls = line.split()
                # image name
                imgs.append('2021VRDL_HW1_datasets/testing_images/'+cls.pop(0))

            self.filenames = imgs  # 資料集的所有檔名

    def __len__(self):

        return len(self.filenames)  # return DataSet 長度

    def __getitem__(self, idx):

        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)  # Transform image

        return self.filenames[idx], image  # return 模型訓練所需的資訊


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
image_size = 456
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

test_transformer = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    normalize
])

model = torch.load('106.pth')
model.eval()

submission = []

fh = open('2021VRDL_HW1_datasets/classes.txt')
dic = []

for line in fh.readlines():
    cls = line.split()
    dic.append(cls.pop(0))

fh = open('2021VRDL_HW1_datasets/testing_img_order.txt')
test_imgs = []

for line in fh.readlines():
    cls = line.split()
    test_imgs.append(cls.pop(0))

testdata = Bird_test_Dataset(
    root='2021VRDL_HW1_datasets/testing_img_order.txt',
    transform=test_transformer, is_train=False)
test_dataloader = torch.utils.data.DataLoader(testdata, batch_size=1)

for i, (image_name, x) in enumerate(test_dataloader):
    # print(test_imgs[i])
    with torch.no_grad():  # 測試階段不需要求梯度
        x = x.to(device)

        output = model(x)  # 將測試資料輸入至模型進行測試

        # print(output)
        _, predicted = torch.max(output.data, 1)
        predict_label = dic[int(predicted[0])]
        submission.append([test_imgs[i], predict_label])

print(len(submission))
np.savetxt('answer.txt', submission, fmt='%s')
