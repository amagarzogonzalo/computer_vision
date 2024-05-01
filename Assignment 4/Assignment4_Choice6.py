# -*- coding: utf-8 -*-
"""Ass4-Choice6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1smZIT6nCuKCTPuwcHLc57bzp9NY7HRWM
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import json
from PIL import Image
import os
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import torchsummary
import torch.nn.init as init
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to the input size for your model
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),  # Convert to tensor for the model
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images (adjust values as per your original normalization)
])

kaggle_to_fmnist_mapping = {
    'Tshirts': 0,
    'Lounge Tshirts' : 0,
    'Jeans': 1,
    'Lounge Pants' : 1,
    'Track Pants': 1,
    'Sweaters': 2,
    'Dresses': 3,
    'Jackets': 4,
    'Rain Jacket': 4,
    'Blazers': 4,
    'Sports Sandals':5,
    'Sandals':5,
    'Shirts': 6,
    'Casual Shoes': 7,
    'Sneakers': 7,
    'Handbags': 8,
    'Backpacks': 8,
    'Laptop Bag':8,
    'Trolley Bag': 8
}



class KaggleFashionDataset(Dataset):
    def __init__(self, dataframe, images_dir, transform=None):
        self.data_list = []
        self.transform = transform
        s = 0
        for idx, row in dataframe.iterrows():
            image_id = str(row['id'])
            image_path = os.path.join(images_dir, image_id + '.jpg')
            if os.path.exists(image_path):
                #self.data_list.append((image_path, row['articleType']))
                if row['articleType'] in kaggle_to_fmnist_mapping:
                    #we append the object to the dataset
                    self.data_list.append((image_path, kaggle_to_fmnist_mapping[row['articleType']]))
            else:
                s+=1

        #print("count: ", s)
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, label = self.data_list[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label

csv_file_path = r"C:\Users\alimo\Desktop\ass4cv\styles.csv"
styles_df = pd.read_csv(csv_file_path, on_bad_lines='skip')
relevant_styles_df = styles_df[styles_df['articleType'].isin(kaggle_to_fmnist_mapping.keys())]


# Create the dataset for testing
images_dir = r"C:\Users\alimo\Desktop\ass4cv\images"

train_df, test_df = train_test_split(styles_df, test_size=0.2, random_state=42)
train_dataset = KaggleFashionDataset(train_df, images_dir,transform)
test_dataset = KaggleFashionDataset(test_df,  images_dir,transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(new_val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
print(len(train_loader),len(test_loader))

# Create the dataset for testing
#kaggle_dataset = KaggleFashionDataset(relevant_styles_df, images_dir, transform)


# BASELINE: LeNet-5 architecture
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            # convolution: 6 out channels/filters = 6@28x28, a convolution of 5x5 is applied -> kernel_size 5
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   # padding 2 because input images are 28x28 (not 32x32)
            nn.ReLU(),
            # average pooling
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            # convolution: it recieves 6 from previous convolution and now 16@10x10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
      return self.classifier(self.feature(x))


# Kaiming Uniform initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

final_model = LeNet5().to(device)

torchsummary.summary(final_model, input_size=(1,28 , 28))

def train(model, device, trainloader, optimizer, criterion, epochs = 5):
  training_loss = 0.0

  model.train()
  correct, total = 0, 0
  for batch_idx, (images, labels) in enumerate(train_loader):
      images, labels = images.to(device), labels.to(device)

      # Forward Pass

      output = model(images)
      loss = criterion(output, labels)


      # Backward Pass
      loss.backward()
      optimizer.step() # updates model parameters using gradient computings by back propagation and applies the optimization algo
      optimizer.zero_grad()


      _, predicted = output.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()

      training_loss += loss.item()
      if batch_idx % 100 == 99:
          print(f'Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}')

  training_accuracy = 100 * correct / total
  average_training_loss = training_loss / len(train_loader)
  print(f"Train accuracy: {training_accuracy} %, Average Loss: {average_training_loss}")
  return training_accuracy, average_training_loss


def test(data_loader, model, loss_fn, type_test):
  with torch.no_grad():
      correct, total , loss = 0, 0, 0
      num_batches = len(data_loader)
      model.eval()
      for images, labels in data_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          loss += loss_fn(outputs, labels.long() ).item()

          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  loss /= num_batches
  accuracy = 100*correct/total
  print(f"{type_test} accuracy: {accuracy} %, Average Loss: {loss}")
  return accuracy, loss


def final_test(dataloader_train, dataloader_test, final_model, final_optimizer, final_criterion, epochs=15):
  training_losses = []
  training_accuracies = []
  for t in range(epochs):
      print(f"Epoch {t+1}\n---------------------")
      training_accuracy, training_loss = train(final_model, device, train_loader, final_optimizer, final_criterion)
      training_losses.append(training_loss)
      training_accuracies.append(training_accuracy)
  accuracy_test, loss_test = test(dataloader_test, final_model, final_criterion, "Test")
  return accuracy_test, loss_test


final_model.apply(init_weights)
final_optimizer = optim.Adam(final_model.parameters(), lr=0.001)
final_criterion = nn.CrossEntropyLoss()

accuracy_test_train, loss_test_train = final_test(train_loader, test_loader, final_model, final_optimizer, final_criterion)




















'''
model = baseline_model
model.eval()

# Create a DataLoader for the Kaggle dataset
kaggle_loader = DataLoader(kaggle_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in kaggle_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Number of relevant styles in dataframe: {len(relevant_styles_df)}")
print(f"Number of items in dataset: {len(kaggle_dataset)}")
print(styles_df['articleType'].unique())
accuracy = correct / total
print(f'Accuracy of the model on the Kaggle dataset: {accuracy * 100}%')
'''