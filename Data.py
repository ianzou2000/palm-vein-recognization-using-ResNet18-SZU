## 划分测试集、训练集
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import os
import re
from tqdm import tqdm
from preprocess import *

train_set = []
train_label = []
test_set = []
test_label = []
train_dataset = []
test_dataset = []

path1 = 'NIR/'
dbtype_list = os.listdir(path1)
dbtype_list.pop(dbtype_list.index('.DS_Store'))

for i in tqdm(range(len(dbtype_list))):
    pic_name = os.listdir(path1+dbtype_list[i])
    for j in range(len(pic_name)):
        if int(pic_name[j][0]) == 1:
            img = cv2.imread(path1 + dbtype_list[i] + '/' + pic_name[j])
            img = hist(img)
            #img = img[45:250, 75:285]
            train_label.append(int(dbtype_list[i])-1)
            train_set.append(pipline(img).numpy().tolist())
        else:
            img = cv2.imread(path1 + dbtype_list[i] + '/' + pic_name[j])
            img = hist(img)
            #img = img[45:250, 75:285]
            test_label.append(int(dbtype_list[i])-1)
            test_set.append(pipline(img).numpy().tolist())
            
print("Data Set has been processed.")

train_dataset = TensorDataset(torch.tensor(train_set), torch.tensor(train_label))  # 对两个列表进行压缩后作为训练集
test_dataset = TensorDataset(torch.tensor(test_set), torch.tensor(test_label))  # 对两个列表进行压缩后作为测试集

TrainLoader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)  # 加载训练集
TestLoader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True)  # 加载测试集