#coding:utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import csv
from models import Attention
from models import CNNAttention
emb_dimension=200
user_count=2580
poi_count=2578
channel=8
net=CNNAttention(user_count,poi_count,emb_dimension,channel)
net.load_state_dict(torch.load('./model/CNNAttentionmodel(canshu)200.pkl'))
#net=Attention(user_count,poi_count,emb_dimension)
#net.load_state_dict(torch.load('./model/Attentionmodel(canshu)200.pkl'))
print("模型初始化完成")
topK=10
memory_rate=0.7
f=open('./data/testset.csv','r')
reader=csv.reader(f)
user={}
for line in reader:
    user.setdefault(line[0],[])
    user[line[0]].append(int(line[4]))
data=[]
for key in user.keys():
    user_id = []
    history_id = []
    label = []
    if len(user[key])<5:
        continue
    user_id.append(int(key))
    for i in range(len(user[key])):
        if i<=(len(line))*memory_rate and i>=(len(line))*memory_rate*0.8:
            history_id.append(int(user[key][i]))
        if i > (len(line)) * memory_rate:
            label.append(int(user[key][i]))
    label=set(label)
    label=list(label)
    if len(label)==0:
        continue
    l = [user_id, history_id, label]

    data.append(l)
print("数据加载完成")
Pre=0
Rec=0
T=0
userchoosed=[]
user_number=len(data)
for line in data:
    count = 0
    if len(line[1])==0:
        user_number-=1
        continue
    y=net(line[0],line[1])
    _, index = torch.topk(y, topK)
    index = index.data.numpy()
    index = index.tolist()
    target=line[2]
    T += len(target)
    target=set(target)
    for i in index:
        for j in target:
            if i==j:
                count+=1
                break
    pre=count/(topK*1.0)
    rec=count/(len(target)*1.0)
    Pre+=pre
    Rec+=rec
    # if pre>0.05 and pre<0.3:
    #     userchoosed.append(str(line[0][0]))
print(user_number)
Precision=Pre/(user_number*1.0)
Recall=Rec/(user_number*1.0)
print(Precision)
print(Recall)
print(userchoosed)
'''
f=open('./data/userchoosed.csv','w')
writer=csv.writer(f)
writer.writerow(userchoosed)
'''
