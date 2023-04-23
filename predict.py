import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import cv2
import os
import numpy as np
import re
from tqdm import tqdm
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义图像增强算法和图像预处理方法

def hist(image):
    b, g, r = cv2.split(image)
    r1 = cv2.equalizeHist(r)
    g1 = cv2.equalizeHist(g)
    b1 = cv2.equalizeHist(b)
    image_equal_clo = cv2.merge([r1, g1, b1])
    return image_equal_clo


pipline = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.Grayscale(3),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
        #transforms.Normalize((0.1307, 0.1307, 0.1307),(0.3081, 0.3081, 0.3081))

    ]
)

# 迁移学习ResNet结构定义

num_classes = 500
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, num_classes)
)

for p in model.parameters():
    p.requires_grad = False
for layer in [model.layer4.parameters(), model.fc.parameters()]:
    for p in layer:
        p.requires_grad = True


model.load_state_dict(torch.load('/Users/ianzou/Desktop/Cornorstone/机器学习/大作业_掌静脉识别/resnet_final.pth', map_location='cpu'))
# model.to(DEVICE)
model.eval()

# 预测集的打包

path1 = '/Users/ianzou/Desktop/Cornorstone/机器学习/大作业_掌静脉识别/new_valdataset/'
dbtype_list = os.listdir(path1)
dbtype_list.pop(dbtype_list.index('.DS_Store'))
pred_label = []
pred_set1 = []
pred_set2 = []
for i in range(len(dbtype_list)):
    if dbtype_list[i][:5] == 'false':
        pred_label.append(0)
    else:
        pred_label.append(1)
    pic_name = os.listdir(path1+dbtype_list[i])
    for j in range(len(pic_name)):
        if pic_name[j][:5] == 'first':
            img = cv2.imread(path1 + dbtype_list[i] + '/' + pic_name[j])
            img = hist(img)
            pred_set1.append(pipline(img).cpu().numpy().tolist())
        else:
            img = cv2.imread(path1 + dbtype_list[i] + '/' + pic_name[j])
            img = hist(img)
            pred_set2.append(pipline(img).cpu().numpy().tolist())

test_set_m1 = TensorDataset(torch.tensor(pred_set1))
test_set_m2 = TensorDataset(torch.tensor(pred_set2))

# 预测主函数

fpr_list = []
tpr_list = []
auc_score = []
suss_list = []

# theta = 0.7
for theta in [0.9921]:
    cor = 0
    wro_acc = 0
    wro_ref = 0
    ref = 0
    real = []
    predict = []
    suss = []
    for i in tqdm(range(len(test_set_m1))):
        t1 = test_set_m1[i][0].to(DEVICE).unsqueeze(0).float()
        t2 = test_set_m2[i][0].to(DEVICE).unsqueeze(0).float()
        t1_output = model(t1)
        t2_output = model(t2)
        t1_compare = t1_output.reshape(t1_output.shape[0]*t1_output.shape[1])
        t1_compare = t1_compare.unsqueeze(0).float()
        t2_compare = t2_output.reshape(t2_output.shape[0]*t2_output.shape[1])
        t2_compare = t2_compare.unsqueeze(0).float()
        simi = torch.cosine_similarity(t1_compare, t2_compare)
        # simi = F.pairwise_distance(t1_compare, t2_compare)
        # simi = np.corrcoef(t1_compare.cpu().detach().numpy(), t2_compare.cpu().detach().numpy())[0][-1]
        # print(simi)
        if simi >= theta:
            if pred_label[i] == 1:
                predict.append(1)
                real.append(1)
                cor += 1
                suss.append(1)
                # if simi >= 13:
                #     print(simi)
            else:
                predict.append(1)
                real.append(0)
                wro_acc += 1
                suss.append(0)
        elif pred_label[i] == 1:
                predict.append(0)
                real.append(1)
                wro_ref += 1
                suss.append(0)
                # if simi >= 0.985:
                #     print(simi)
        else:
            predict.append(0)
            real.append(0)
            ref += 1
            suss.append(1)
    fpr, tpr, thersholds = roc_curve(real, predict)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_score.append(auc(fpr, tpr))
    suss_list.append(np.sum(suss))