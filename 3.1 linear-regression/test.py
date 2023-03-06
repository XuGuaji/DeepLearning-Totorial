# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:36:51 2023

@author: XuDonghui
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim
'''生成数据集'''
num_inputs = 2
num_examples = 1000
true_w = [3,-4]
true_b = -3.5
features = torch.tensor(np.random.normal(0,1,(num_examples,num_inputs)),dtype=torch.float)
labels = features[:,0] * true_w[0] + features[:,1] * true_w[1] + true_b
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)
'''读取数据'''
batch_size = 10
dataset = Data.TensorDataset(features,labels)
data_iter = Data.DataLoader(dataset,batch_size,shuffle=True)
'''定义模型'''
class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet,self).__init__()
        # self.linear调用了nn.Linear的模型，生成n_feature输入1输出的模型
        self.linear = nn.Linear(n_feature,1)
    def forward(self,x):
        y = self.linear(x)
        return y
        
'''
没有forward函数会报错
NotImplementedError:
    Module [LinearNet] is missing the required "forward" function
'''
net = LinearNet(num_inputs)
'''
LinearNet(
  (linear): Linear(in_features=2, out_features=1, bias=True)
)
'''

'''初始化模型参数'''
init.normal_(net.linear.weight,mean=0,std=0.01)
init.constant_(net.linear.bias, val=0)
'''定义损失函数'''
loss = nn.MSELoss()
'''定义优化算法'''
optimizer = optim.SGD(net.parameters(),lr=0.03)
'''
SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    lr: 0.03
    maximize: False
    momentum: 0
    nesterov: False
    weight_decay: 0
)
'''
'''训练模型'''
num_epochs = 3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        output = net.forward(X)
        # output = net(X)
        l = loss(output,y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print("epoch %d,loss %f"%(epoch,l.item()))
    print(net.linear.weight,net.linear.bias)
# num_epochs = 3
# for epoch in range(1, num_epochs + 1):
#     for X, y in data_iter:
#         # output = net.forward(X)#和下面等价
#         output = net(X)
#         l = loss(output, y.view(-1, 1))
#         optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
#         l.backward()
#         optimizer.step()
#     print('epoch %d, loss: %f' % (epoch, l.item()))













