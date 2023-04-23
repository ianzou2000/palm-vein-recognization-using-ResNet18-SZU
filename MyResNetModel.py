## 迁移学习ResNet结构定义
from preprocess import *

num_classes = 495
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, num_classes)
)
model = model.to(DEVICE)

for p in model.parameters():
    p.requires_grad = False
for layer in [model.layer4.parameters(), model.fc.parameters()]:
    for p in layer:
        p.requires_grad = True

params_non_frozen = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(params_non_frozen, lr=5*1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 设定优优化器更新的时刻表
