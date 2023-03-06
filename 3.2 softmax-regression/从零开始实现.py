# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:39:05 2023

@author: XuDonghui
"""
import torch
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms

'''读取数据'''
batch_size = 256

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
# 很奇怪,不可以多线程读取
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
'''
feature,labels = mnist_train[0]

labels
Out[4]: 9

feature.size()
Out[8]: torch.Size([1, 28, 28])
'''

'''初始化模型参数'''
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

'''实现softmax运算'''
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1,keepdim=True)
    return X_exp / partition

'''定义模型'''
def net(X):
    return softmax(torch.mm(X.view((-1,num_inputs)),W)+b)

'''定义损失函数'''
def cross_entropy(y_hat,y):
    return - torch.log(y_hat.gather(1,y.view(-1,1)))
# net(X)返回的是什么格式的y

'''计算分类准确率'''
def accuracy(y_hat,y):
    return (y_hat.argmax(dim=1)==y).float().mean().item()

'''评价模型net在数据集data_iter上的准确率'''
def evaluate_accuracy(data_iter,net):
    acc_sum,n = 0.0,0
    for X,y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
'''定义优化算法'''
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data


'''训练模型'''
num_epochs, lr = 5, 0.1

def train_softmax(net,train_iter,test_iter,loss,num_epochs,batch_size,
                  params = None, lr = None, optimizer = None):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n = .0,.0,0
        for X,y in train_iter:
            y_hat = net(X)
            # X: 256 * 784
            # y_hat: 256 * 10
            # y: 256 * 1
            l = loss(y_hat,y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd(params,lr,batch_size)
            else:
                optimizer.step()
        
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d,loss %.4f,train_acc %.3f,test acc %.3f"
              %(epoch+1,train_l_sum / n,train_acc_sum / n,test_acc))

train_softmax(net,train_iter,test_iter,cross_entropy,num_epochs,batch_size,
              params = [W,b], lr = 0.1, optimizer = None)