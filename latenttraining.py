#coding:utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import csv
from models import LRmodel
emb_dimension=200
user_count=2580
poi_count=2578
net=LRmodel(user_count,poi_count,emb_dimension,5)
print("成功初始化模型")
data=[]
file1=open('./data/traindata12.csv','r')
reader=csv.reader(file1)
for line in reader:
    l=[]
    user_id=[]
    poi_id=[]
    C=[]
    for i in range(len(line)):
        if i==0:
            user_id.append(int(line[i]))
            continue
        if i==1:
            poi_id.append(int(line[i]))
            continue
        else:
            C.append(int(line[i]))
    l = [user_id, poi_id,C]
    data.append(l)
file2=open('./data/negativesample12.csv','r')
reader=csv.reader(file2)
count=0
for line in reader:
    neg=[]
    for i in range(0,len(line)):
        neg.append(int(line[i]))
    data[count].append(neg)
    count+=1
file2.close()
print("训练数据加载完成")
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
total_loss=0
for i in range(len(data)):
    if i%1000==999:
        print(total_loss/1000)
        total_loss=0
    optimizer.zero_grad()
    loss=net.forward(data[i][0],data[i][1],data[i][2],data[i][3])
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
print("训练完成")
filename='./model/LRmodel(canshu)'+str(emb_dimension)+'_12'+'.pkl'
torch.save(net.state_dict(), filename)
