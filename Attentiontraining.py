#coding:utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import csv
from models import Attention
from models import ResAttention
from models import CNNAttention
from models import LRmodel
import random
'''
Channels=[32,64,128]
for channel in Channels:
    emb_dimension=200
    user_count=2580
    poi_count=2578
    #channel=32
    memory_rate=0.7
    net=CNNAttention(user_count,poi_count,emb_dimension,channel)
    LRnet=LRmodel(user_count,poi_count,emb_dimension,5)
    LRnet.load_state_dict(torch.load('./model/LRmodel(canshu)200.pkl'))
    net.UserPreference.weight=LRnet.UserPreference.weight
    net.PoiPreference.weight=LRnet.PoiPreference.weight
    print("模型初始化完成")
    f=open('./data/attentiondata.csv','r')
    reader=csv.reader(f)
    data=[]
    reader=csv.reader(f)
    for line in reader:
        if len(line)<3:
            continue
        l=[]
        user_id=[]
        history_id=[]
        label=[]
        for i in range(len(line)):
            if i==0:
                user_id.append(int(line[i]))
                continue
            if i<=(len(line)-1)*memory_rate:
                history_id.append(int(line[i]))
                continue
            else:
                label.append(int(line[i]))
        l = [user_id, history_id,label]
        data.append(l)
    f.close()
    print("数据加载完成")
    criterion = nn.BCELoss()
    UEmb_params=list(map(id,net.UserPreference.parameters()))
    PEmb_params=list(map(id,net.PoiPreference.parameters()))
    base_params=filter(lambda p: id(p) not in UEmb_params+PEmb_params,net.parameters())
    params = [
        {"params": net.UserPreference.parameters(), "lr": 0.001},
        {"params": net.PoiPreference.parameters(), "lr": 0.001},
        {"params": base_params, "lr": 0.025},
    ]
    optimizer = optim.SGD(params,momentum=0.9)
    total_loss=0.0
    for epoch in range(10):
        random.shuffle(data)
        for i in range(len(data)):
            if i%1000==999:
                print(total_loss/1000)
                total_loss=0
            optimizer.zero_grad()
            output=net.forward(data[i][0],data[i][1])
            label = torch.zeros(poi_count)
            target=data[i][2]
            for i in target:
                label[i]=1.0
            loss = criterion(output, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    print("训练完成")
    filename='./model/CNNAttentionmodel(canshu)'+str(emb_dimension)+'_'+str(channel)+'.pkl'
    torch.save(net.state_dict(), filename)
'''
emb_dimension=200
user_count=2580
poi_count=2578
channel=8
memory_rate=0.7
net=CNNAttention(user_count,poi_count,emb_dimension,channel)
LRnet=LRmodel(user_count,poi_count,emb_dimension,5)
LRnet.load_state_dict(torch.load('./model/LRmodel(canshu)200_12.pkl'))
net.UserPreference.weight=LRnet.UserPreference.weight
net.PoiPreference.weight=LRnet.PoiPreference.weight
print("模型初始化完成")
f=open('./data/attentiondata.csv','r')
reader=csv.reader(f)
data=[]
reader=csv.reader(f)
for line in reader:
    if len(line)<3:
        continue
    l=[]
    user_id=[]
    history_id=[]
    label=[]
    for i in range(len(line)):
        if i==0:
            user_id.append(int(line[i]))
            continue
        if i<=(len(line)-1)*memory_rate:
            history_id.append(int(line[i]))
            continue
        else:
            label.append(int(line[i]))
    l = [user_id, history_id,label]
    data.append(l)
f.close()
print("数据加载完成")
criterion = nn.BCELoss()
UEmb_params=list(map(id,net.UserPreference.parameters()))
PEmb_params=list(map(id,net.PoiPreference.parameters()))
base_params=filter(lambda p: id(p) not in UEmb_params+PEmb_params,net.parameters())
params = [
    {"params": net.UserPreference.parameters(), "lr": 0.001},
    {"params": net.PoiPreference.parameters(), "lr": 0.001},
    {"params": base_params, "lr": 0.025},
]
optimizer = optim.SGD(params,momentum=0.9)
total_loss=0.0
for epoch in range(10):
    random.shuffle(data)
    for i in range(len(data)):
        if i%1000==999:
            print(total_loss/1000)
            total_loss=0
        optimizer.zero_grad()
        output=net.forward(data[i][0],data[i][1])
        label = torch.zeros(poi_count)
        target=data[i][2]
        for i in target:
            label[i]=1.0
        loss = criterion(output, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
print("训练完成")
filename='./model/CNNAttentionmodel(canshu)'+str(emb_dimension)+'_'+str(channel)+'_12'+'.pkl'
torch.save(net.state_dict(), filename)
