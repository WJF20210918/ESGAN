import os
import math
import time
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torch.utils.data as Data
from MyExperiments.Networks.VGG_16 import Vgg16
from MyExperiments.Networks.AlexNet import AlexNet
from MyExperiments.Networks.ResNet_18 import ResNet18

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
        img = (np.array(img)).astype('uint8')  # 将数据存进内存
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


# SR模型
class EDSR(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()

        encoder_lis = [
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=True),
        ]

        bottle_neck_lis = [ResnetBlock(64) for _ in range(8)]
        bottle_neck_lis.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True))

        upsampler_lis = [
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True),
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.upsampler = nn.Sequential(*upsampler_lis)

    def forward(self, x):
        x = self.encoder(x)
        res = self.bottle_neck(x)
        res += x
        x = self.upsampler(res)
        return x


# 噪声模型
class Noise(nn.Module):
    def __init__(self):
        super(Noise, self).__init__()

        encoder_lis = [
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=True),  # 112
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=True),  # 110
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # 55
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),  # 55
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        ]

        bottle_neck_lis = [ResnetBlock(256, True),
                           ResnetBlock(256, True),
                           ResnetBlock(256, True),
                           ResnetBlock(256, True), ]

        decoder_lis = [
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 110
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0, bias=False),  # 112
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 224
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),  # 224
            nn.Tanh()
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


# 生成模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.edsr = EDSR()
        self.noise = Noise()
        self.conv = nn.Conv2d(6, 3, 1, 1, 0)

    def forward(self, x):
        pert = self.noise(x)
        sr = self.edsr(x)
        str = torch.cat([pert, sr], dim=1)
        str = self.conv(str)
        # str = sr  + pert
        return str


# 判别模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        ]
        self.model = nn.Sequential(*model)
        self.fc = nn.Linear(256 * 14 * 14, 1)

    def forward(self, x):
        out = self.model(x)
        output = torch.sigmoid(self.fc(out.contiguous().view(x.shape[0], -1)))  # .contiguous()
        return output


# 残差块
class ResnetBlock(nn.Module):
    def __init__(self, dim, bn=False):
        super(ResnetBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(dim, dim, 3, 1, 1))
            if bn: m.append(nn.BatchNorm2d(dim))
            if i == 0: m.append(nn.ReLU(True))
        self.conv_block = nn.Sequential(*m)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# 评估准确率
def evaluate_accuracy(net, strG, dir, batch_size):
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
    one, two, three, four, five = 0, 0, 0, 0, 0
    c_one, c_two, c_three, c_four, c_five = 0, 0, 0, 0, 0
    strG.eval()
    with torch.no_grad():
        for step in range(steps_test):
            test_data = MyDataset(step, batch_size, name, image, label)
            test_iter = Data.DataLoader(test_data, batch_size, num_workers=0,
                                        shuffle=False,
                                        pin_memory=True, )
            X = []
            y = []
            N = []
            for names, img, lab in test_iter:
                X = img
                y = lab
                N = names
            X = torch.transpose(X, 1, 3)  # N C W H
            X = torch.transpose(X, 2, 3)  # N C H W
            X = X.to(device)
            y = y.to(device)

            str_X = strG(X)

            # save images
            # im = torch.transpose(noise, 2, 3)
            # im = torch.transpose(im, 1, 3)  # N H W C
            # out = im.clone().cpu().detach().numpy()
            # for i in range(batch_size):
            #     i_out = out[i, ]
            #     i_out = np.array(i_out) * 255
            #     i_out = np.clip(i_out, 0., 255.)
            #     im = np.uint8(i_out)
            #     im = Image.fromarray(im.astype('uint8'), mode='RGB')
            #     im.save(dir_image + N[i])
            #     np.save('F:/MyExperiment/enhanced_images/' + N[i].split('.')[0] + '.npy', i_out)

            for i in range(batch_size):
                if net(str_X[i][None, :, :, :]).argmax(dim=1) != 0:  # true->wrong
                    if y[i] == 0:
                        one += 1
                    c_one += 1
                if net(str_X[i][None, :, :, :]).argmax(dim=1) != 1:
                    if y[i] == 1:
                        two += 1
                    c_two += 1
                if net(str_X[i][None, :, :, :]).argmax(dim=1) != 2:
                    if y[i] == 2:
                        three += 1
                    c_three += 1
                if y[i] == 0:  # true->wrong
                    if net(str_X[i][None, :, :, :]).argmax(dim=1) != 0:
                        five += 1
                    c_five += 1
                # elif net(str_X[i][None, :, :, :]).argmax(dim=1) != 1:
                #     if y[i] == 1:
                #         two += 1
                #     c_two += 1
                # elif net(str_X[i][None, :, :, :]).argmax(dim=1) != 2:
                #     if y[i] == 2:
                #         three += 1
                #     c_three += 1
                # elif net(str_X[i][None, :, :, :]).argmax(dim=1) != 3:
                #     if y[i] == 3:
                #         four += 1
                #     c_four += 1
                # else:  # wrong->true
                #     if y[i] == 4:
                #         five += 1
                #     c_five += 1

            # acc_sum += (net(str_X).argmax(dim=1) == y).float().sum().cpu().item()
            n += y.shape[0]

    print('one=', one, 'two=', two, 'three=', three, 'four=', four, 'five=', five)
    print('c_one=', c_one, 'c_two=', c_two, 'c_three=', c_three, 'c_four=', c_four, 'c_five=', c_five)
    strG.train()
    return acc_sum / n


# 训练模型
def train_process(net, strG, strD, dir_train, dir_test,
                  batch_size, num_epochs, optimizer_G, optimizer_D):
    print("training on ", device)
    list_train = os.listdir(dir_train)
    steps_train = math.floor(float(len(list_train)) / batch_size)

    for epoch in range(num_epochs):
        loss_D_sum = 0.0
        loss_G_sum = 0.0
        loss_y_sum = 0.0
        loss_str_sum = 0.0
        train_acc, test_acc, n, start = 0.0, 0.0, 0, time.time()
        for step in range(steps_train):
            train_data = MyDataset(step, batch_size, train_name, train_img, train_label)
            train_iter = Data.DataLoader(train_data, batch_size, num_workers=0,
                                         shuffle=True,
                                         pin_memory=True, )
            X = []
            y = []
            for _, image, label in train_iter:
                X = image
                y = label
            X = torch.transpose(X, 1, 3)  # N C W H
            X = torch.transpose(X, 2, 3)  # N C H W
            org_X = X.to(device)
            y = y.long().to(device)

            str_X = strG(org_X)

            # 训练判别器----------------------------------------
            optimizer_D.zero_grad()
            pred_real = strD(org_X).squeeze()
            loss_D_real = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real, device=device))
            pred_fake = strD(str_X.detach()).squeeze()
            loss_D_fake = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake, device=device))
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()

            # 训练生成器-------------------------------------------
            optimizer_G.zero_grad()
            # loss_G_fake  使生成样本与原始样本接近(对抗约束)
            pred_fake = strD(str_X).squeeze()
            loss_G_fake = F.binary_cross_entropy(pred_fake, torch.ones_like(pred_fake, device=device))
            loss_G_fake.backward(retain_graph=True)

            # loss_y  目标模型损失(类别约束)
            y_pert = net(str_X)
            loss_y = F.cross_entropy(y_pert, y)

            # loss_str  像素约束
            loss_str = torch.dist(str_X, org_X, p=2)  # 计算欧式距离(L2范数)

            y_lambda = 100
            str_lambda = 0.01
            loss_G = y_lambda * loss_y + str_lambda * loss_str
            loss_G.backward()
            optimizer_G.step()

            loss_D_sum += loss_D.cpu().item()
            loss_G_sum += loss_G_fake.cpu().item()
            loss_y_sum += loss_y.cpu().item()
            loss_str_sum += loss_str.cpu().item()
            n += y.shape[0]

        train_acc = evaluate_accuracy(net, strG, dir_train, batch_size)
        test_acc = evaluate_accuracy(net, strG, dir_test, batch_size)
        print('epoch %d, loss_D: %.3f, loss_G: %.3f, loss_y: %.3f, '
              'loss_str: %.3f, train_acc %.3f, test_acc %.3f, time %.1f sec'
              % (epoch + 1, loss_D_sum / n, loss_G_sum / n, loss_y_sum / n,
                 loss_str_sum / n, train_acc, test_acc, time.time() - start))


batch_size = 20
lr, num_epochs = 0.0001, 120
dir_train = 'F:/MyExperiment/ImageNetTwo/adv_train_A'
dir_test = 'F:/MyExperiment/ImageNetThree/adv_test_A'
dir_save = 'F:/MyExperiment/pretrained_models/esgan_alexnet_3(120).pth'  # 模型保存路径
# dir_image = 'F:/MyExperiment/ImageNetThree/enhanced_images_R_noise/'

train_name, train_img, train_label = load_data(dir_train)
test_name, test_img, test_label = load_data(dir_test)

net = AlexNet().to(device)
net.load_state_dict(torch.load('F:/MyExperiment/pretrained_models/alexnet_model_3.pth'))
net.eval()  # 设为评估模式

# net = Vgg16().to(device)
# net.load_state_dict(torch.load('F:/MyExperiment/pretrained_models/vgg16_model_2.pth'))
# net.eval()  # 设为评估模式

# net = ResNet18().to(device)
# net.load_state_dict(torch.load('F:/MyExperiment/pretrained_models/resnet18_model_3.pth'))
# net.eval()  # 设为评估模式

strG = Generator().to(device)
strD = Discriminator().to(device)
optimizer_G = torch.optim.Adam(strG.parameters(), lr=lr)  # 生成优化器
optimizer_D = torch.optim.Adam(strD.parameters(), lr=lr)  # 判别优化器

# train_process(net, strG, strD, dir_train, dir_test, batch_size, num_epochs, optimizer_G, optimizer_D)
# torch.save(strG.state_dict(), dir_save)  # 保存模型参数
start = time.time()
strG.load_state_dict(torch.load(dir_save))  # 加载模型
accuracy = evaluate_accuracy(net, strG, dir_test, batch_size)
print('Test accuracy %.3f, time %.1f sec' % (accuracy, time.time()-start))
