import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
epochs = 200
number_of_batches = 16
learning_rate = 0.01



# Data preparation

class CustomImageDataset(Dataset):
    def __init__(self, labels_arr, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.DataFrame(labels_arr)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_dataset(folder, label):
    files = os.listdir(folder)
    for i in files:
        if i == '.DS_Store':
            files.remove(i)
    files = np.array(files)
    labels_arr = np.zeros(len(files), dtype=np.uint8)
    labels_arr.fill(label)
    labels_arr = np.column_stack((files, labels_arr))

    transform = transforms.Compose([transforms.Resize((360, 640))])
    dataset = CustomImageDataset(labels_arr, folder, transform=transform)
    return dataset


low = r"C:\Users\yakovgay\Downloads\gaky_shit\low"
medium = r"C:\Users\yakovgay\Downloads\gaky_shit\medium"
high = r"C:\Users\yakovgay\Downloads\gaky_shit\high"
off = r"C:\Users\yakovgay\Downloads\gaky_shit\off"

low_blur = r"C:\Users\yakovgay\Downloads\gaky_shit\low blur"
medium_blur = r"C:\Users\yakovgay\Downloads\gaky_shit\medium blur"
high_blur = r"C:\Users\yakovgay\Downloads\gaky_shit\high blur"
off_blur = r"C:\Users\yakovgay\Downloads\gaky_shit\off blur"


dataloader_low = create_dataset(low, int(0))
dataloader_medium = create_dataset(medium, int(1))
dataloader_high = create_dataset(high, int(2))
dataloader_off = create_dataset(off, int(3))

dataloader_low_blur = create_dataset(low_blur, int(0))
dataloader_medium_blur = create_dataset(medium_blur, int(1))
dataloader_high_blur = create_dataset(high_blur, int(2))
dataloader_off_blur = create_dataset(off_blur, int(3))


#currently we have 964 images in dataset
train_dev_sets = ConcatDataset([dataloader_low, dataloader_medium, dataloader_high,
                                dataloader_off, dataloader_low_blur, dataloader_medium_blur, dataloader_high_blur,
                                dataloader_off_blur])


dataloader = DataLoader(train_dev_sets, batch_size=64, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 60, 3, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(60, 600, 3, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(600, 1200, 3, stride=1)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(600 * 90 * 160, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)  # -> n, 10
        return x


model = ConvNet().to(device)

loss_function = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        model.zero_grad()
        out = model(x)

        loss = loss_function(out, y)
        loss = loss.mean()
        loss.backward()
        optim.step()

        if epoch % 5 == 0:
            #logic for saving model
            pass

        if i % 5 == 0:
            cat = torch.argmax(out, dim=1)
            accuracy = (cat == y).float().mean()
            print(f'Epoch: {epoch}, Accuracy: {accuracy}')
