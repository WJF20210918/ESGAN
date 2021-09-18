import time
import os
import math
import torch
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 读取数据
def load_data(dir):
    list = os.listdir(dir)
    listname = []
    listimg = []
    listlabel = []
    for i in range(len(list)):
        img = Image.open(dir + '/' + list[i])
        img = img.convert('RGB')
        img = (np.array(img)).astype('uint8')
        listimg.append(img)
        listlabel.append(int(list[i].split('_')[1]))
        listname.append(list[i])
    return listname, listimg, (np.array(listlabel)).astype('uint8')


class MyDataset(Data.Dataset):
    def __init__(self, step, batch, name, img, lab):
        self.name = name[step * batch: (step + 1) * batch]
        self.img = img[step * batch: (step + 1) * batch]
        self.lab = lab[step * batch: (step + 1) * batch]
        self.len = batch

    def __getitem__(self, index):
        name = self.name[index]
        img = self.img[index]
        image = img.astype(np.float32) / 255.
        lab = self.lab[index]
        return name, image, lab

    def __len__(self):
        return self.len


# 定义模型
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)  # in, out, kernel_size, stride, padding
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3_bn = nn.BatchNorm2d(512)

        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
            # nn.Linear(7*7*512, 4096),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(4096, 4096),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(4096, 3),
        )

    def forward(self, img):
        h = F.relu(self.conv1_1(img))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # 112

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2_bn(self.conv2_2(h)))
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # 56

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # 28

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # 14

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3_bn(self.conv5_3(h)))
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # 7

        output = self.fc(h.view(img.shape[0], -1))
        return torch.softmax(output, dim=1)


# 评估准确率
def evaluate_accuracy(net, dir, batch_size):
    print("testing on ", device)
    if dir == dir_train:
        name = train_name
        image = train_img
        label = train_label
    else:
        name = test_name
        image = test_img
        label = test_label

    steps_test = math.floor(float(len(label)) / batch_size)

    acc_sum, n = 0.0, 0
    net.eval()  # 评估模式, 这会关闭dropout
    with torch.no_grad():
        for step in range(steps_test):
            test_data = MyDataset(step, batch_size, name, image, label)
            test_iter = Data.DataLoader(test_data, batch_size, num_workers=0,
                                        shuffle=False,
                                        pin_memory=True, )
            N = []
            X = []
            y = []
            for names, img, lab in test_iter:
                N = names
                X = img
                y = lab
            X = torch.transpose(X, 1, 3)  # N C W H
            X = torch.transpose(X, 2, 3)  # N C H W
            X = X.to(device)
            y = y.to(device)

            # for i in range(batch_size):
            #     img = X[i][None, :]
            #     im = torch.transpose(img, 2, 3)
            #     im = torch.transpose(im, 1, 3)  # N H W C
            #     out = im.clone().cpu().detach().numpy()
            #     i_out = out[0, ]
            #     i_out = np.array(i_out) * 255.
            #     i_out = np.clip(i_out, 0., 255.)
            #     im = np.uint8(i_out)
            #     im = Image.fromarray(im.astype('uint8'), mode='RGB')
            #     if net(img).argmax(dim=1) == y[i]:
            #         im.save('E:/MyExperiment/ImageNetTwo/true_V/' + N[i])
            #     else:
            #         im.save('E:/MyExperiment/ImageNetTwo/wrong_V/' + N[i])

            acc_sum += (net(X).argmax(dim=1) == y).float().sum().cpu().item()
            n += y.shape[0]

    net.train()  # 改回训练模式
    return acc_sum / n


# 训练模型
def train_process(net, dir_train, dir_test, batch_size, num_epochs, optimizer, Ir_scheduler):
    print("training on ", device)
    list_train = os.listdir(dir_train)
    steps_train = math.floor(float(len(list_train)) / batch_size)

    for epoch in range(num_epochs):
        train_loss, n, start = 0.0, 0, time.time()
        for step in range(steps_train):
            train_data = MyDataset(step, batch_size, train_name, train_img, train_label)
            train_iter = Data.DataLoader(train_data, batch_size, num_workers=0,
                                         shuffle=True,
                                         pin_memory=True,)
            X = []
            y = []
            for _, img, label in train_iter:
                X = img
                y = label
            X = torch.transpose(X, 1, 3)  # N C W H
            X = torch.transpose(X, 2, 3)  # N C H W
            X = X.to(device)
            y = y.long().to(device)

            y_hat = net(X)
            l = F.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss += l.cpu().item()
            n += y.shape[0]

        train_acc = evaluate_accuracy(net, dir_train, batch_size)
        test_acc = evaluate_accuracy(net, dir_test, batch_size)
        print('epoch %d, train_loss %.4f, train_acc %.3f, test_acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss / n, train_acc, test_acc, time.time() - start))

        # Ir_scheduler.step(epoch)


batch_size = 50
lr, num_epochs = 0.0001, 5
dir_save = 'E:/MyExperiment/pretrained_models/vgg16_model_2.pth'  # 模型保存路径
dir_train = 'E:/MyExperiment/ImageNetTwo/train'
dir_test = 'E:/MyExperiment/ImageNetTwo/test'

train_name, train_img, train_label = load_data(dir_train)
test_name, test_img, test_label = load_data(dir_test)

# net = Vgg16().to(device)
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 优化器
# Ir_scheduler = StepLR(optimizer, step_size=2, gamma=0.95)  # 学习率衰减

# train_process(net, dir_train, dir_test, batch_size, num_epochs, optimizer, Ir_scheduler)
# torch.save(net.state_dict(), dir_save)  # 保存模型参数
# net.load_state_dict(torch.load(dir_save))  # 加载模型参数
# start = time.time()
# test_acc = evaluate_accuracy(net, dir_test, batch_size)
# print('test_acc %.3f, time %.1f sec' % (test_acc, time.time()-start))
