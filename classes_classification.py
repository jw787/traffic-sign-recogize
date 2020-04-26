#! /usr/bin/env python
# coding=utf-8

import copy
import os
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data.sampler import WeightedRandomSampler

ROOT_DIR = '../traffic_sign/'
TRAIN_DIR = 'train/'
TEST_DIR = 'test/'
TRAIN_ANNO = 'Classes_train_annotation.csv'
TEST_ANNO = 'Classes_test_annotation.csv'
NUM_CLASSES = 62


class My_dataset():

    def __init__(self, root_dir, anno_file, transform=None):

        self.root_dir = root_dir
        self.anno_file = anno_file
        self.transform = transform

        if not os.path.isfile(self.anno_file):
            print(self.anno_file + 'does not exist !')
        # sep=','   # 以，为数据分隔符
        # shkiprows= 10   # 跳过前十行
        # nrows = 10   # 只去前10行
        # parse_dates = ['col_name']   # 指定某行读取为日期格式
        # index_col = ['col_1','col_2']   # 读取指定的几列
        # error_bad_lines = False   # 当某行数据有问题时，不报错，直接跳过，处理脏数据时使用
        # na_values = 'NULL'   # 将NULL识别为空值
        # 这里的index_col=0对去指定的第一列为索引值
        self.file_info = pd.read_csv(anno_file, index_col=0)
        # 无法用debug来显示是因为行数太多了
        # print(self.file_info)

    # __len__，使得len(dataset)返回数据集的大小
    def __len__(self):
        return len(self.file_info)

    # __getitem__支持索引，使得dataset[i]可以用来获取第i个样本, 就是为了处理不同类的样本中每个图片的功能区域
    def __getitem__(self, idx):
        # 这里代表了self.file_info是DataFrame形式，可用df['image_location']形式获取列名为'image_location'的列的所有data，
        # idx用来获取第idx个样本
        img_path = self.file_info['path'][idx]
        if not os.path.isfile(img_path):
            print(img_path + 'does not exist!')
            return None
        # 为了保证图片输出通道顺序为RGB模式
        image = Image.open(img_path).convert('RGB')
        # 获取图像的类别信息
        label_class = int(self.file_info.iloc[idx]['classes'])
        # 这里的样本表示形式用字典更加容易接下来的使用
        sample = {'image': image, 'classes': label_class}
        # 这里的样本表示形式用字典更加容易接下来的使用
        if self.transform:
            # transform 可以处理字典形式
            sample['image'] = self.transform(image)
            return sample


train_transforms = transforms.Compose([transforms.Resize((224, 224)),  # PIL独有形式image
                                       transforms.RandomRotation(10),
                                       transforms.ToTensor(),
                                       ])
test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor()
                                      ])

# 用于获取经过数据增强和处理的Train的Dataset
train_dataset = My_dataset(root_dir=ROOT_DIR + TRAIN_DIR, anno_file=TRAIN_ANNO, transform=train_transforms)
# 用于获取经过数据增强和处理的Test的Dataset
test_dataset = My_dataset(root_dir=ROOT_DIR + TEST_DIR, anno_file=TEST_ANNO, transform=test_transforms)
# histogram(a,bins=10,range=None,weights=None,density=False);
#
#     a是待统计数据的数组；
#
#     bins指定统计的区间个数；
#
#     range是一个长度为2的元组，表示统计范围的最小值和最大值，默认值None，表示范围由数据的范围决定
#
#     weights为数组的每个元素指定了权值,histogram()会对区间中数组所对应的权值进行求和
#
#     density为True时，返回每个区间的概率密度；为False，返回每个区间中元素的个数
#
# a = np.random.rand(100)
# hist,bins = np.histogram(a,bins=5,range=(0,1))
# print(hist)
# print(bins)
#
#
# 输出：
#
# [19 30 15 16 20]
# [ 0.   0.2  0.4  0.6  0.8  1. ]
# 原文链接：https://blog.csdn.net/hankobe1/java/article/details/104931766
# def classes_num():
#     class_num = pd.read_csv('Classes_train_annotation.csv')['classes']
#     class_num = np.array([class_num], dtype=np.int)
#     class_num = class_num.reshape(-1)
#     return class_num


# class_num = classes_num()
# # print(class_num)
# # a = train_dataset[1]
# # print(a)
# train_class_weight, _ = np.histogram(class_num, bins=NUM_CLASSES, range=(0, NUM_CLASSES), density=True)
# print(train_class_weight)
# train_class_weight = torch.from_numpy(train_class_weight)
# 用于处理和整理Dataset，并放入DataLoader中 out od memory是可一把batch_size调整小一些
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# train_loader = DataLoader(train_dataset,batch_size=256, pin_memory=True, num_workers=4,
#                           sampler=WeightedRandomSampler(train_class_weight,NUM_CLASSES))
test_loader = DataLoader(test_dataset)
# 这里是为了下面根艺方便编写代码把整个数据写成字典形式
data_loader = {'train': train_loader, 'test': test_loader}
# 这里为用gpu加速的固定格式写法
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# 随机可视化一些数据图片
def imshow():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset
    # 随机获取一个样本图片
    sample = sample[idx]
    # 显示索引图片的值，和图片的维度，类别号码的信息
    print(idx, sample['image'].shape, sample['classes'])
    # 样本图片的地址信息
    sample_image_path = sample['image']
    # 将其棉花为PIL对去西施，再对去图片路径，纪委用pil的根式去后区图片的路径，再把PIL的地址形式显示出来
    plt.imshow(transforms.ToPILImage()(sample_image_path))
    plt.show()


imshow()


def train_model(model, criterion, optimizer, num_epoch=32):
    # 记录和计算运行时间
    since = time.time()
    # 为了下面添加loss值和accuracy的值提供“篮子”
    loss_list = {'train': [], 'test': []}
    class_acc_list = {'train': [], 'test': []}
    # —–我们寻常意义的复制就是深复制，即将被复制对象完全再复制一遍作为独立的新个体单独存在。所以改变原有被复制对象不会对已经复制出来的新对象产生影响。
    # —–而浅复制并不会产生一个独立的对象单独存在，他只是将原有的数据块打上一个新标签，所以当其中一个标签被改变的时候，数据块就会发生变化，
    # 另一个标签也会随之改变。这就和我们寻常意义上的复制有所不同了。**
    best_model_wts = copy.deepcopy(model.state_dict())  # 只保存网络中的参数 (速度快, 占内存少)
    best_acc = 0.0  # 初始化最好模型的准确率
    # 为了显示实时训练轮数和总共所需训练轮数
    for epoch in range(num_epoch):
        print(f'Epoch： {epoch}/{num_epoch - 1}')
        # 为了分隔面次轮数的训练情况，便于区分
        print('_*' * 20)

        # Each epoch has a training and testing phase， 这里为了遍历train和test中的数据集,操作不一致

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            # 初始化损失值和被正确分类的类型个数，以便下面用到是可以使用
            running_loss = 0.0
            correct_classes = 0.0

            for idx, data in enumerate(data_loader[phase]):
                # print(phase + f' processing: {idx}th batch.')
                # 数据导入是从My_dataset过来的，所以索引到他的样本为sample={'image':image, 'classes':labels_classes}
                # 但是因为DataLoader的原因，他输入的图片每一轮的数量为batch_size
                inputs = data['image'].to(device)
                labels_classes = data['classes'].to(device)
                class_num = len(data['image'])
                train_class_weight, _ = np.histogram(np.array(data['image']), bins=class_num, range=(0, class_num),
                                                     density=True)
                train_class_weight = torch.from_numpy(train_class_weight).to(device)
                # zero the parameter gradients，# 清空上一步的残余更新参数值，因为pytorch会有参数保存机制，
                # 根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
                # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad了
                # 即为了避免数据集每过一轮后的梯度保存着并且再下一轮计算梯度时相加
                optimizer.zero_grad()
                # forward
                # track history if only in train
                # 这里是为了当phase=="train"，他的bool值为true，且当他的值为true是，下面的程序才可自动求导
                with torch.set_grad_enabled(phase == 'train'):
                    x_classes = model(inputs)
                    # 为了抱着输出格式为(n,NUM_CLASSES),因为有62类
                    x_classes = x_classes.view(-1, NUM_CLASSES)
                    # torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引），这里纪委class_id
                    _, pred_classes = torch.max(x_classes, 1)

                    loss = criterion(x_classes, labels_classes)
                    # 这里运用不同权重来计算每一个的loss值
                    loss = (loss * train_class_weight).mean()

                    # 对于训练集需要进行参数优化，而测试集则无此必要
                    if phase == 'train':
                        # 梯度反传
                        loss.backward()
                        # optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,但是不绝对，可以根据具体的需求来做。
                        # 只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整。通常我们有
                        #
                        # optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
                        # scheduler = lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
                        # model = net.train(model, loss_function, optimizer, scheduler, num_epochs = 100)
                        # 原文链接：https://blog.csdn.net/qq_20622615/java/article/details/83150963
                        # 梯度更新
                        optimizer.step()
                # 这里的每一次循环为inputs_size(0)=64，因为每一轮就输入64张图片，并且因为DataLoader的原因，他会以每次batchsize为64进行一次循环，
                # 直到遍历全部数据后完成循环，所以这里的loss.item()具体就是： 用于将一个零维张量转换成浮点数，因为这里的loss为平均loss
                running_loss = loss.item() * inputs.size(0)
                # 计算正确的数目，当且仅当预测值和真实值为true是，他才加1，这里是遍历到以64个为一单位，#计算所有batch的精确度
                correct_classes += torch.sum(pred_classes == labels_classes)
            # 这里len(data_loader[phase].dataset)为数据集中的数据数量
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            # 获取由损失值组成的列表
            loss_list[phase].append(epoch_loss)

            epoch_acc_classes = correct_classes.double() / len(data_loader[phase].dataset)
            epoch_acc = epoch_acc_classes
            # 这里是为了获取每一轮的准确率
            class_acc_list[phase].append(100 * epoch_acc_classes)
            print(f'{phase} Loss: {epoch_loss:.4f}  Acc_classes: {epoch_acc_classes:.2%}')
            # 处理样本总数得到平均的loss
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc_classes
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'Best val classes Acc: {best_acc:.2%}')
    # 导入目前最好的模型的参数
    model.load_state_dict(best_model_wts)
    # 保存模型的参数，一文件名为‘best_model.pt'形式存储再当前目录下，  torch.save(state, filepath)
    torch.save(model.state_dict(), 'best_model.pt')
    # 获取进行程序的结束时间
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # 输出最佳准确率
    print(f'Best test classes Acc: {best_acc:.2%}')

    return model, loss_list, class_acc_list


# 因为Pytorch有着autograd即自动求导机制，因为我们不想自动求导，所以把各层的权重给固定住
# Convnet作为固定特征提取器，只训练最后一层，通过require_grad=False冻结vgg16_bn的早期层

# 迁移学习
# 微调卷积神经网络，以rvgg16_bn为例进行微调
# model_conv = models.vgg16_bn(pretrained=True)
model_conv = models.resnet18(pretrained=True)
# model_conv = models.wide_resnet50_2(pretrained=True)
# model_conv = models.resnet34(pretrained=True)
# print(model_conv)
for param in model_conv.parameters():
    param.requires_grad = False
# 这届用模型跑
# # model_conv = models.vgg16_bn(pretrained=True)
# model_conv = models.resnet18()
# # model_conv = models.wide_resnet50_2(pretrained=True)
# # model_conv = models.resnet34(pretrained=True)
# # print(model_conv)

# #最后fc层的输入
# num_ftrs = model_conv.classifier[6].in_features
num_ftrs = model_conv.fc.in_features
# 修改全连接层，并保留前面所有层，进行优化训练， 并且将最后的线性输出改为类别个数
model_conv.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_conv = model_conv.to(device)

# 这里用单目标多分类的交叉熵损失函数
# criterion = nn.CrossEntropyLoss(reduction=None)
# 这里使用的是带权重的采样来计算损失
criterion = nn.CrossEntropyLoss()
# 选择最流行的优化器Adam优化器
optimizer = optim.Adam(model_conv.parameters(), lr=0.01)
# 训练
model, loss_list, class_acc_list = train_model(model_conv, criterion, optimizer, num_epoch=10)
# print(loss_list['train'])

# 显示预测效果,即准确值和损失值的却邪趋势

y1 = loss_list["test"]
y2 = loss_list["train"]
x = range(0, len(y1))

plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="test")
plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
plt.legend()
plt.title('train and test loss vs. epoches')
plt.ylabel('loss')
plt.savefig("train and test loss vs epoches.png")
plt.close('all')  # 关闭图 0

y5 = class_acc_list["train"]
y6 = class_acc_list["test"]
plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="test")
plt.legend()
plt.title('train and test Classes_acc vs. epoches')
plt.ylabel('Classes_accuracy')
plt.savefig("train and test Classes_acc vs epoches.png")
plt.close('all')


############################################ Visualization ###############################################
def visualize_model(model):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader['test']):
            inputs = data['image']
            labels_classes = data['classes'].to(device)

            x_classes = model(inputs.to(device))
            x_classes = x_classes.view(-1, NUM_CLASSES)
            _, preds_classes = torch.max(x_classes, 1)

            print(inputs.shape)
            plt.imshow(transforms.ToPILImage()(inputs.squeeze(0)))
            plt.title(f'predicted classes: {preds_classes}\n ground-truth classes:{labels_classes}')
            plt.show()


visualize_model(model)
