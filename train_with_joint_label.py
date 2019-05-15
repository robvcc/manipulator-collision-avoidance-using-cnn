from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from ResNet import *
import os
from tensorboardX import SummaryWriter

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 500               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 30
LR = 0.001
# LR = 0.001              # learning rate


root = "./"


def default_loader(path):
    return Image.open(path).convert('RGB')
    #return Image.open(path)


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            value = words[1].split(",")
            label = []
            for item in value:
                label.append(float(item))

            imgs.append((words[0], label))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        fh.close()

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        # img = Image.fromarray(np.array(img), mode='L')
        if self.transform is not None:
            img = self.transform(img)
            # label = self.transform(label)
        return img, label

    def __len__(self):
        return len(self.imgs)


train_data = MyDataset(txt=root + 'label/joint_train.txt', transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = MyDataset(txt=root + 'label/joint_test.txt', transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        # self.out = nn.Linear(32 * 7 * 7, 5)   # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 50 * 75, 6)   # fully connected layer, output 10 classes
        # self.out = nn.Linear(32 * 320 * 180, 5)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


# cnn = CNN()
# cnn = torchvision.models.resnet50(pretrained=False, num_classes=6)
cnn = resnet50(pretrained=False, num_classes=6)
if os.path.exists('./model/joint_net_params.pkl'):  # 加载训练过的模型
   cnn.load_state_dict(torch.load('./model/joint_net_params.pkl'))

cnn.cuda()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss(reduction='mean')

writer = SummaryWriter(log_dir='./logs')

print(len(train_data), len(train_loader))
# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        sum_loss = 0
    # -----------------------------------------------------
        label = []
        for item in y:
            label.append(item.numpy())
        label = np.array(label)
        label = label.astype(np.float32)
        label = label.T
        y = torch.tensor(label)
        # print(type(y), y.dtype, y.size())
    # -----------------------------------------------------
        b_x = Variable(x).cuda()  # batch x
        b_y = Variable(y).cuda()  # batch y

        # print(type(b_y), b_y.dtype, b_y.size())
        # output = cnn(b_x)[0]  # cnn output
        output = cnn(b_x)
        # print(type(output), output.dtype, output.size())

        # print(output)
        loss = loss_func(output, b_y)  # cross entropy loss
        sum_loss += loss.item()
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 30 == 0:
            cnn.eval()
            eval_loss = 0.
            eval_acc = 0.
            for i, (tx, ty) in enumerate(test_loader):
                # --------------------------------------------------txl add---------
                label = []
                for item in ty:
                    label.append(item.numpy())
                label = np.array(label)
                label = label.astype(np.float32)
                label = label.T
                ty = torch.tensor(label)
                # print(type(ty), ty.dtype, ty.size())
                # ---------------------------------------------------

                t_x = Variable(tx).cuda()
                t_y = Variable(ty).cuda()
                output = cnn(t_x)
                # output = cnn(t_x)[0]
                loss = loss_func(output, t_y)
                eval_loss += loss.item()
                # ------------------------------------------------------txl add ---------
                out = output.cpu().data.numpy()
                y = t_y.cpu().data.numpy()

                for k in range(len(out)):
                    count = 0
                    for j in range(len(out[k])):
                        if np.abs(out[k, j]-y[k, j]) <= 0.1:
                            count = count + 1
                    if count == len(out[k]):
                        eval_acc = eval_acc + 1
                # -----------------------------------------------------
            acc_rate = eval_acc / float(len(test_data))
            print(len(test_data), eval_acc)
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)), acc_rate))

            x = 1 if step == 0 else step * 30
            writer.add_scalar('train_loss', sum_loss / x, epoch * len(train_loader) + step)
            writer.add_scalar('test_loss', eval_loss / (len(test_data)), epoch * len(train_loader) + step)
            writer.add_scalar('acc_rate', acc_rate, epoch * len(train_loader) + step)
writer.close()
torch.save(cnn.state_dict(), "./model/joint_net_params.pkl")