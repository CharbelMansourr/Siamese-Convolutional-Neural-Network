#Loading the Dataset from the google drive
from google.colab import drive
drive.mount('/content/drive/')


%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps   
from google.colab.patches import cv2_imshow 
import os
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from skimage import io

#Loading the images from the folders class
class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDatasets,transform=None, Train =True):
            
        self.transform = transform
        self.images_list = []
        self.labels_list = []
        self.imgFolderDatasets=imageFolderDatasets
        self.typee="Test"
        if Train:
          self.typee="Train"
        
        labels = os.listdir(os.path.join(imageFolderDatasets,self.typee))
        
        if len(labels)>2:
          labels[0]=labels[2]
          labels[1]=labels[1]   
          labels.pop()
        self.dirs = labels 
        
        #print(labels)
        self.imageFolderDataset = os.listdir(os.path.join(imageFolderDatasets,self.typee))

        i = 0
        for label in labels:
          files = os.listdir(os.path.join(imageFolderDatasets,self.typee,label))
          self.images_list += files
          self.labels_list += [i]*len(files)
          i += 1

    def __getitem__(self,idx):
      if self.labels_list[idx]==0:
        folder=self.dirs[0]
      else:
        folder=self.dirs[1]
      image1_path=os.path.join(self.imgFolderDatasets,self.typee,folder,self.images_list[idx])
      idx1 = np.random.randint(len(self.images_list))

      if self.labels_list[idx1]==0:
        folder2=self.dirs[0]
      else: 
        folder2=self.dirs[1]
      image2_path=os.path.join(self.imgFolderDatasets,self.typee,folder2,self.images_list[idx1])

      img1 = Image.open(image1_path)
      img2 = Image.open(image2_path)
      img1 = img1.convert("L")
      img2 = img2.convert("L")

      label1 = self.labels_list[idx]
      label2 = self.labels_list[idx1]
      if label1 == label2:
        label = 1
      else:
        label = 0

      if self.transform is not None:
          img1 = self.transform(img1)
          img2 = self.transform(img2)
      return img1,img2,label

    def __len__(self):
        return len(self.images_list)



class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU()
        )

        # Setting up the Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(),          
            nn.Linear(1024, 256),
            nn.ReLU(),           
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # The output is used to determine the similiarity
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive


# Resize the images and transform to tensors
transformation = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])

# Initialize the network
siamese_dataset = SiameseNetworkDataset("/content/drive/MyDrive/Newdata",transform=transformation)


train_loader = DataLoader(siamese_dataset,shuffle=True,num_workers=2,batch_size=2)

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0005 )


counter = []
loss_history = [] 
iteration_number= 0

# Iterate throught the epochs
for epoch in range(100):

    # Iterate over batches
    for  i, data in enumerate(train_loader,0):

        # Send the images and labels to CUDA
        img0, img1, label = data[0].cuda(), data[1].cuda(), data[2].cuda()

        # Zero the gradients
        optimizer.zero_grad()

        # Pass in the two images into the network and obtain two outputs
        output1, output2 = net(img0, img1)

        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = criterion(output1, output2, label)

        # Calculate the backpropagation
        loss_contrastive.backward()

        # Optimize
        optimizer.step()

        # Every 10 batches print out the loss
        if i % 10 == 0 :
            #print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
            iteration_number += 10
        counter.append(iteration_number)
        loss_history.append(loss_contrastive.item())


plt.plot(counter,loss_history)
plt.show()


siamese_dataset = SiameseNetworkDataset("/content/drive/MyDrive/Newdata",transform=transformation,Train=False)
test_dataloader = DataLoader(siamese_dataset, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
# Grab one image that we are going to test
x0, _, _ = next(dataiter)

for i in range(10):
    # Iterate over 10 images x1 and test them with the first image (x0)
    _, x1, label2 = next(dataiter)

    # Concatenate the two images together
    concatenated = torch.cat((x0, x1), 0)
    
    output1, output2 = net(x0.cuda(), x1.cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)

    xx=torchvision.utils.make_grid(concatenated)
    im = transforms.ToPILImage()(xx).convert("RGB")
    print(f'Dissimilarity: {euclidean_distance.item():.2f}' )
    display(im)