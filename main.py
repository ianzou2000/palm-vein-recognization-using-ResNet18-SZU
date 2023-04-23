from MyResNetModel import *
from Data import *
from function import *

## 模型运行主函数
train_loss = []
train_acc = []
test_acc = []
top5_acc = []

for epoch in range(1, EPOCHS + 1):
    #scheduler.step()
    if epoch == 15:
        train_loss = train_model(model, optimizer, epoch, DEVICE, TrainLoader, train_loss)
        test_acc = test(model, DEVICE, TestLoader, test_acc)
        top5_acc = evaluteTop5(model, DEVICE, TestLoader, top5_acc)
        print(train_loss, train_acc, test_acc, top5_acc)
    train_model(model, optimizer, epoch, DEVICE, TrainLoader, train_loss)
    test(model, DEVICE, TestLoader, test_acc)
    evaluteTop5(model, DEVICE, TestLoader, top5_acc)

torch.save(model.state_dict(), './resnet_test.pth')