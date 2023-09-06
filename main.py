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
batch_size = 32
learning_rate = 0.05
momentum = 0.2


#Data preparation
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
        label = torch.tensor(int(self.img_labels.iloc[idx, 1]))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    
def create_dataset(folder, label):
    files = os.listdir(folder)
    for i in files:
        if i =='.DS_Store':
            files.remove(i)
    files = np.array(files)
    labels_arr = np.zeros(len(files), dtype=np.uint8)
    labels_arr.fill(label)
    labels_arr = np.column_stack((files, labels_arr))

    transform = transforms.Compose([transforms.Resize((360, 640))])
    dataset = CustomImageDataset(labels_arr, folder, transform=transform)
    return dataset

dataloader_low = create_dataset("/Users/gaky/Desktop/efir/low", int(0))
dataloader_medium = create_dataset("/Users/gaky/Desktop/efir/medium", int(1))
dataloader_high = create_dataset("/Users/gaky/Desktop/efir/high_small", int(2))
dataloader_off = create_dataset("/Users/gaky/Desktop/efir/off", int(3))

dataloader_low_blur = create_dataset("/Users/gaky/Desktop/efir/low blur", int(0))
dataloader_medium_blur = create_dataset("/Users/gaky/Desktop/efir/medium blur", int(1))
dataloader_high_blur = create_dataset("/Users/gaky/Desktop/efir/high_blur_small", int(2))
dataloader_off_blur = create_dataset("/Users/gaky/Desktop/efir/off blur", int(3))

train_dev_sets = ConcatDataset([dataloader_low, dataloader_medium, dataloader_high, 
                                dataloader_off, dataloader_low_blur, dataloader_medium_blur, dataloader_high_blur, dataloader_off_blur])

dataloader = DataLoader(train_dev_sets, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(train_dev_sets, batch_size=900, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3, stride=1, padding=1)#360x640
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.pool1 = nn.MaxPool2d(2, stride=2)#180x320
        
        self.conv2 = nn.Conv2d(20, 30, 3, stride=1, padding=1)#180x320
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        
        self.pool2 = nn.MaxPool2d(2, stride=2)#90x160
        
        self.conv3 = nn.Conv2d(30, 30, 3, stride=1, padding=1)#90x160
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        
        self.pool3 = nn.MaxPool2d(2, stride=2)#45x80
        
        self.conv4 = nn.Conv2d(30, 30, 3, stride=1, padding=1)#45x80
        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
        
        self.pool4 = nn.MaxPool2d(2, stride=2)#22x40
        
        self.conv5 = nn.Conv2d(30, 30, 3, stride=1, padding=1)#22x40
        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
        
        self.pool5 = nn.MaxPool2d(2, stride=2)#11*20
        

        self.fc1 = nn.Linear(30*11*20, 100)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        
        self.fc2 = nn.Linear(100, 30)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        
        self.fc3 = nn.Linear(30, 4)
        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.pool2(x)
            
        x = self.conv3(x)
        x = F.relu(x)
        

        x = self.pool3(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = F.relu(x)
        
        x = self.pool5(x)
        
        
        x = x.view(-1, 30*11*20)   
        
        
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        
        x = self.fc3(x)
        return x


model = ConvNet().to(device)

loss_function = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


for epoch in range(epochs):
    for i, (x, y) in enumerate(dataloader):
        
        x = x/255.
        x = x.to(device)
        y = y.to(device)
        
        model.zero_grad()
        x = F.normalize(x.float())
        out = model(x.float())

        loss = loss_function(out, y.long())
        loss = loss.mean()
        loss.backward()
        optim.step()

        if epoch % 5 == 0:
            #logic for saving model
            pass
        
        if epoch % 30 == 0 and i==5:
            dataset_test = iter(dataloader_test)
            x, y = next(dataset_test)
            x = x/255.
            x = x.to(device)
            y = y.to(device)
            
            with torch.no_grad():
                out = model(x.float())
                cat = torch.argmax(out, dim=1)
                accuracy = (cat == y.long()).float().mean()
                print("accuracy", accuracy)
                print(out)
                print(y.long())
                
        print(epoch, i)
        #print(f'Epoch: {epoch}, Accuracy: {accuracy}', y, cat)
