import torch
import torch.nn.functional as F

## 模型训练函数定义

def train_model(model, optimizer, epoch, device, TrainLoader, train_loss):
    model.train()
    accuracy = 0.0
    for (batch_index, data) in enumerate(TrainLoader):
        x_data, label = data
        x_data = x_data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(x_data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        predicted = torch.max(output.data,1)[1]
        accuracy += (predicted == label).sum()
        if batch_index % 900 == 0:
            print("Train Epoch:{} \nTrain -- Loss:{:.6f}, Accuracy:{:.3f}".format(epoch, loss.item(), float(accuracy*100)/float(8)*(batch_index+1)))
            train_loss.append(loss.item())
    return train_loss

## 模型测试函数定义

def test(model, device, TestLoader, test_acc):
    model.eval()
    accuracy = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for (batch_index, data) in enumerate(TestLoader):
            y_data, label = data
            y_data = y_data.to(device)
            label = label.to(device)
            output = model(y_data)
            test_loss += F.cross_entropy(output, label).item()
            pred = output.max(1, keepdim=True)[1]
            accuracy += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(TestLoader.dataset)
        print("Test -- Average loss:{:.4f}, Top-1 Accuracy:{:.3f}".format(test_loss, 100.0*accuracy/len(TestLoader.dataset)))
        test_acc.append(100.0*accuracy/len(TestLoader.dataset))
    return test_acc

def evaluteTop5(model, device, TestLoader, top5_acc):
    model.eval()
    correct = 0
    total = len(TestLoader.dataset)
    for X, y in TestLoader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            logits = model(X)
            maxk = max((1, 5))
            y_resize = y.view(-1, 1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, y_resize).sum().float().item()
    print("Test -- Top-5 Accuracy:{:.3f}\n".format(correct / total * 100.0))
    top5_acc.append(correct / total * 100.0)
    return top5_acc
