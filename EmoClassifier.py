import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

class CustomDataset(Dataset):
    def __init__(self,path):
        self.imgs_path = path
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
        self.class_map = {"angry" : 0, "disgust":1, "fear":2, "happy":3, "neutral": 4, "sad":5, "surprise":6}
        self.img_dim = (64, 64)    
    def __len__(self):
        return len(self.data)    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id

train_dataset = CustomDataset("MMAFEDB/train/")
train_data_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

train_data_loader_accuracy = DataLoader(train_dataset,batch_size=1000, shuffle=True)
test_dataset = CustomDataset("MMAFEDB/test/")
test_data_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

val_dataset = CustomDataset("MMAFEDB/valid/")
val_data_loader = DataLoader(train_dataset, batch_size=1000,shuffle=True)



class FaceEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img=nn.Sequential(
            nn.Conv2d(3,64,3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3,3),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,3,stride=1),
            nn.ReLU(),
            nn.Conv2d(128,256,3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3,3),
            nn.Flatten(),
            nn.Linear(5*5*256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,7),
            nn.BatchNorm1d(7),
        )
    def forward(self,X):  
        return self.img(X.float())

model=FaceEmotionModel().to(device)
cost_fn = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epoch=20
batch_size=100
costs=[]
s=time.time()

def predict(X,y):
    correct=0
    model.eval()
    y_pred = model.forward(X)
    pred=F.softmax(y_pred,dim=1)
    pred=torch.argmax(pred,axis=1)
    correct=(y == pred).sum()
    return correct


X_train_acc,y_train_acc=next(iter(train_data_loader_accuracy))
X_val_acc,y_val_acc=next(iter(val_data_loader))
for i in tqdm(range(1,epoch+1)):
    total_cost=0
    for X,y in tqdm(train_data_loader):
        optimizer.zero_grad()
        y_pred=model(X.to(device))
        loss = cost_fn(y_pred, y.squeeze().to(device))
        loss.backward()
        optimizer.step()
        total_cost+=loss.item()
    costs.append(total_cost/batch_size)
    print("Cost after epoch "+str(i)+" = "+str((total_cost/batch_size)))
    X_train_acc,y_train_acc=next(iter(train_data_loader_accuracy))
    y_train_acc=y_train_acc.squeeze()
    X_val_acc,y_val_acc=next(iter(val_data_loader))
    y_val_acc=y_val_acc.squeeze()
    print("Accuracy on Train set is "+str(np.array(predict(X_train_acc.to(device),y_train_acc.to(device)).cpu())/y_train_acc.size(0)))
    
    print("Accuracy on Validation set is "+str(np.array(predict(X_val_acc.to(device),y_val_acc.to(device)).cpu())/y_val_acc.size(0)))
    
X_test_acc,y_test_acc=next(iter(test_data_loader))
y_test_acc=y_test_acc.squeeze()
print("Accuracy on Test set is "+str(np.array(predict(X_test_acc.to(device),y_test_acc.to(device)).cpu())/y_test_acc.size(0)))


print("Training time taken: "+str(time.time()-s))
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('epochs')
plt.show()

#torch.save(model.state_dict(),"trained.pth")


