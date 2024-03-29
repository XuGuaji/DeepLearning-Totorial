# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:22:48 2023

@author: XuDonghui
"""
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

'''生成数据集'''
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

'''读取数据'''
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

'''定义模型'''
# class LinearNet(nn.Module):
#     def __init__(self, n_feature):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(n_feature, 1)
#     # forward 定义前向传播
#     def forward(self, x):
#         y = self.linear(x)
#         return y
class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    def forward(self,x):
        y = self.linear(x)
        return y
net = LinearNet(num_inputs)
# net = nn.Sequential(
#     nn.Linear(num_inputs, 1)
#     # 此处还可以传入其他层
#     )
'''初始化模型参数'''
init.normal_(net.linear.weight,mean=0,std=0.01)
init.constant_(net.linear.bias, val=0)
# init.normal_(net[0].weight, mean=0, std=0.01)
# init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

'''定义损失函数'''
loss = nn.MSELoss()

'''定义优化算法'''
optimizer = optim.SGD(net.parameters(), lr=0.03)


'''训练模型'''
num_epochs = 3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        output = net(X)
        l = loss(output,y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print("epoch %d,loss: %f"%(epoch,l.item()))

dense = net.linear
print(dense.weight)
print(dense.bias)