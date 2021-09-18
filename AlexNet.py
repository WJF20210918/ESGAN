import time
import os
import math
import torch
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
import torch.utils.data as Data

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
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        # 这⾥全连接层使⽤丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
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
            #         im.save('E:/MyExperiment/ImageNetTwo/true_A/' + N[i])
            #     else:
            #         im.save('E:/MyExperiment/ImageNetTwo/wrong_A/' + N[i])

            acc_sum += (net(X).argmax(dim=1) == y).float().sum().cpu().item()
            n += y.shape[0]

    net.train()  # 改回训练模式
    return acc_sum / n


# 训练模型
def train_process(net, dir_train, dir_test, batch_size, num_epochs, optimizer):
    print("training on ", device)
    list_train = os.listdir(dir_train)
    steps_train = math.floor(float(len(list_train)) / batch_size)

    for epoch in range(num_epochs):
        train_l_sum, n, start = 0.0, 0, time.time()
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

            y_pre = net(X)
            loss = F.cross_entropy(y_pre, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l_sum += loss.cpu().item()
            n += y.shape[0]

        train_acc = evaluate_accuracy(net, dir_train, batch_size)
        test_acc = evaluate_accuracy(net, dir_test, batch_size)
        print('epoch %d, train_loss %.4f, train_acc %.3f, test_acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc, test_acc, time.time() - start))


batch_size = 50
lr, num_epochs = 0.0001, 20
dir_save = 'E:/MyExperiment/pretrained_models/alexnet_model_2.pth'  # 模型保存路径
dir_train = 'E:/MyExperiment/ImageNetTwo/train'
dir_test = 'E:/MyExperiment/ImageNetTwo/adv_train_A'

train_name, train_img, train_label = load_data(dir_train)
test_name, test_img, test_label = load_data(dir_test)

# net = AlexNet().to(device)
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# train_process(net, dir_train, dir_test, batch_size, num_epochs, optimizer)
# torch.save(net.state_dict(), dir_save)  # 保存模型参数
# net.load_state_dict(torch.load(dir_save))  # 加载模型参数
# start = time.time()
# test_acc = evaluate_accuracy(net, dir_test, batch_size)
# print('test_acc %.3f, time %.1f sec' % (test_acc, time.time()-start))

