import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import cv2
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from PIL import Image
import os
import copy
from tqdm import tqdm
import wandb
import argparse

################################################################################################################
#Read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--v', default="v1",help="which config file version to use",type=str)
parser.add_argument('--w',default=1,type=int, help="1 if you want to use wandb")
args = parser.parse_args()
print("version: ", args.v)
print("device: ",args.device)
print("wandb: ",args.w)

import json

config_file_dir = "/home/jatin-pg/MTP/cfgs"
config_file = os.path.join(config_file_dir,args.v+".json")
with open(config_file) as f:
    config_dict = json.load(f)
    model_dict = config_dict["model"]
    data_dict = config_dict["data"]
modes = config_dict["modes"]
data_type = data_dict["type"]
plt.ion()
data_dir = os.path.join("/home/jatin-pg/MTP/PlantVillage-Dataset/raw",data_type)

label_dict = {'Tomato___Spider_mites Two-spotted_spider_mite': 0, 'Peach___Bacterial_spot': 1, 'Cherry_(including_sour)___healthy': 2, 'Pepper,_bell___healthy': 3, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 4, 'Tomato___Leaf_Mold': 5, 'Peach___healthy': 6, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 7, 'Potato___Early_blight': 8, 'Grape___healthy': 9, 'Tomato___healthy': 10, 'Blueberry___healthy': 11, 'Apple___Black_rot': 12, 'Strawberry___Leaf_scorch': 13, 'Corn_(maize)___Common_rust_': 14, 'Tomato___Bacterial_spot': 15, 'Grape___Esca_(Black_Measles)': 16, 'Pepper,_bell___Bacterial_spot': 17, 'Cherry_(including_sour)___Powdery_mildew': 18, 'Corn_(maize)___healthy': 19, 'Tomato___Early_blight': 20, 'Tomato___Septoria_leaf_spot': 21, 'Soybean___healthy': 22, 'Raspberry___healthy': 23, 'Potato___healthy': 24, 'Potato___Late_blight': 25, 'Grape___Black_rot': 26, 'Apple___healthy': 27, 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 28, 'Strawberry___healthy': 29, 'Orange___Haunglongbing_(Citrus_greening)': 30, 'Apple___Cedar_apple_rust': 31, 'Corn_(maize)___Northern_Leaf_Blight': 32, 'Tomato___Tomato_mosaic_virus': 33, 'Tomato___Target_Spot': 34, 'Tomato___Late_blight': 35, 'Apple___Apple_scab': 36, 'Squash___Powdery_mildew': 37}

if args.w==1:
    wandb.init(project='Plant_disease_detection',name=args.v)
    config = wandb.config
    config.learning_rate = model_dict["learning_rate"]
    config.optimizer = model_dict["optimizer"]
    config.num_epochs = model_dict["num_epochs"]

##############################################################################################################
#Create Dataset
class PlantVillageDataset(Dataset):
    """PlantVillage dataset."""

    def __init__(self, file_path, root_dir,label_dict, transform=None):
        """
        Args:
            file_path (string): Path to the txt file which contains image paths
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file = open(file_path,"r+")
        self.paths = self.file.readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.label_dict = label_dict
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.paths[idx].strip("\n")
        label = [x for x in path.split("/")][0]
        label_no = self.label_dict[label]
        img_name = os.path.join(self.root_dir,path)
        image = Image.open(img_name).convert('RGB')
        #image = torch.from_numpy(image).type(torch.FloatTensor)
        #image = image.permute(2,0,1)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label_no, 'label_name': label}
        return sample

#define Dataloaders
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30,30)),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_files = {'train': "/home/jatin-pg/MTP/PlantVillage-Dataset/trainlist.txt",
                'test': "/home/jatin-pg/MTP/PlantVillage-Dataset/testlist.txt",
              'val': "/home/jatin-pg/MTP/PlantVillage-Dataset/vallist.txt"}
image_datasets = {x: PlantVillageDataset(image_files[x],data_dir,label_dict,data_transforms[x])
                  for x in modes}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=4)
              for x in modes}
dataset_sizes = {x: len(image_datasets[x]) for x in modes}
device = torch.device(args.device)


##########################################################################################################
#Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25,model_path=""):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in modes:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for sample in tqdm(dataloaders[phase]):
                inputs = sample['image'].to(device)
                labels = sample['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if args.w==1:
                wandb.log({phase+"_loss": epoch_loss},step = epoch)
                wandb.log({phase+"_accuracy": epoch_acc},step = epoch)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#########################################################################################################

#Define Model
if model_dict["name"]=="resnet34":
    model_ft = models.resnet34(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 38.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, model_dict["num_classes"])

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=model_dict["learning_rate"], momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_path = os.path.join(model_dict["model_dir"],args.v+".pth")

############################################################################################################
#train model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=model_dict["num_epochs"],model_path = model_path)







