import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
class LRmodel(nn.Module):
    def __init__(self,user_count,POI_count,emb_dimension,neg_num,GPU_flag=False):
        super(LRmodel,self).__init__()
        self.emb_dimension=emb_dimension
        self.UserPreference=nn.Embedding(user_count,emb_dimension,sparse=True)
        self.PoiPreference=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.GPU_flag=GPU_flag
        self.init_emb()
    def init_emb(self):
        nn.init.xavier_normal(self.UserPreference.weight)
        nn.init.xavier_normal(self.PoiPreference.weight)
    def forward(self,userid,poii,Ci,neg_p):
        loss=[]
        if(self.GPU_flag):
            emb_u=self.UserPreference(Variable(torch.LongTensor(userid)).cuda())
            emb_pi=self.PoiPreference(Variable(torch.LongTensor(poii)).cuda())
            neg_emb=self.PoiPreference(Variable(torch.LongTensor(neg_p)).cuda())
            if len(Ci):
                emb_Ci=self.PoiPreference(Variable(torch.LongTensor(Ci)).cuda())
        else:
            emb_u = self.UserPreference(Variable(torch.LongTensor(userid)))
            emb_pi = self.PoiPreference(Variable(torch.LongTensor(poii)))
            neg_emb = self.PoiPreference(Variable(torch.LongTensor(neg_p)))
            if len(Ci):
                emb_Ci = self.PoiPreference(Variable(torch.LongTensor(Ci)))
        score=torch.mul(emb_u,emb_pi)
        score = score.sum()
        score=F.sigmoid(score)
        neg_score=torch.mul(emb_u,neg_emb)
        neg_score=neg_score.sum(1)
        neg_score=F.sigmoid(neg_score)
        if len(Ci):
            emb_Ci=emb_Ci.sum(0)
            tmp=torch.mul(emb_Ci,emb_pi)
            tmp=tmp.sum()
            tmp=tmp/float(len(Ci))
            tmp=F.sigmoid(tmp)
            score=torch.mul(score,tmp)
            #score=0.5*score+0.5*tmp
            tmp=torch.mul(emb_Ci,neg_emb)
            tmp=tmp.sum(1)
            tmp = tmp / float(len(Ci))
            tmp=F.sigmoid(tmp)
            neg_score=torch.mul(neg_score,tmp)
            #neg_score=0.5*neg_score+0.5*tmp
        score=torch.log(score)
        neg_score=torch.log(1-neg_score)
        return -(score+neg_score.sum())

class Attention(nn.Module):
    def __init__(self,user_count,POI_count,emb_dimension,GPU_flag=False):
        super(Attention, self).__init__()
        self.emb_dimension = emb_dimension
        self.UserPreference = nn.Embedding(user_count, emb_dimension, sparse=True)
        self.PoiPreference = nn.Embedding(POI_count, emb_dimension, sparse=True)
        self.L1=nn.Linear(emb_dimension,1000)
        self.L2 = nn.Linear(1000, POI_count)
        self.GPU_flag = GPU_flag
        self.init_emb()
    def init_emb(self):
        nn.init.xavier_normal(self.UserPreference.weight)
        nn.init.xavier_normal(self.PoiPreference.weight)
    def forward(self, userid,H):
        if self.GPU_flag:
            emb_u=self.UserPreference(torch.LongTensor(userid).cuda())
            emb_h = self.PoiPreference(Variable(torch.LongTensor(H)).cuda())
        else:
            emb_u = self.UserPreference(Variable(torch.LongTensor(userid)))
            emb_h = self.PoiPreference(Variable(torch.LongTensor(H)))
        a=torch.mul(emb_u,emb_h)
        a=a.sum(1)
        a=a/math.sqrt(self.emb_dimension)
        a=F.softmax(a)
        a=a.view(1,-1)
        output=torch.mm(a,emb_h)
        output=output.view(1,-1)
        output=self.L1(output)
        output=self.L2(output)
        output=F.sigmoid(output)
        output=output.view(-1)
        return output

class ResAttention(nn.Module):
    def __init__(self,user_count,POI_count,emb_dimension,GPU_flag=False):
        super(ResAttention, self).__init__()
        self.emb_dimension = emb_dimension
        self.UserPreference = nn.Embedding(user_count, emb_dimension, sparse=True)
        self.PoiPreference = nn.Embedding(POI_count, emb_dimension, sparse=True)
        self.L1=nn.Linear(emb_dimension,1000)
        self.L2 = nn.Linear(1000, POI_count)
        self.GPU_flag = GPU_flag
        self.init_emb()
    def init_emb(self):
        nn.init.xavier_normal(self.UserPreference.weight)
        nn.init.xavier_normal(self.PoiPreference.weight)
    def forward(self, userid,H):
        if self.GPU_flag:
            emb_u=self.UserPreference(torch.LongTensor(userid).cuda())
            emb_h = self.PoiPreference(Variable(torch.LongTensor(H)).cuda())
        else:
            emb_u = self.UserPreference(Variable(torch.LongTensor(userid)))
            emb_h = self.PoiPreference(Variable(torch.LongTensor(H)))
        a=torch.mul(emb_u,emb_h)
        a=a.sum(1)
        a=a/math.sqrt(self.emb_dimension)
        a=F.softmax(a)
        a=a.view(1,-1)
        output=torch.mm(a,emb_h)
        output=output+emb_h.sum(0)#add residual connection
        output=output.view(1,-1)
        output=self.L1(output)
        output=self.L2(output)
        output=F.sigmoid(output)
        output=output.view(-1)
        return output

class CNNAttention(nn.Module):
    def __init__(self,user_count,POI_count,emb_dimension,channel,GPU_flag=False):
        super(CNNAttention, self).__init__()
        self.emb_dimension = emb_dimension
        self.UserPreference = nn.Embedding(user_count, emb_dimension, sparse=True)
        self.PoiPreference = nn.Embedding(POI_count, emb_dimension, sparse=True)
        self.channel=channel
        self.C1 = nn.Conv2d(1,self.channel,(1,1))
        self.C2 = nn.Conv2d(1,self.channel,(2,1))
        self.C3 = nn.Conv2d(1,self.channel,(4,1))
        self.C4 = nn.Conv2d(1, self.channel, (8, 1))
        self.UC = nn.Conv2d(1,self.channel,(1,1))
        self.L1=nn.Linear(self.emb_dimension*self.channel,1000)
        self.L2 = nn.Linear(1000, POI_count)
        self.GPU_flag = GPU_flag
        self.init_emb()
    def init_emb(self):
        nn.init.xavier_normal(self.UserPreference.weight)
        nn.init.xavier_normal(self.PoiPreference.weight)
    def forward(self, userid,H):
        if self.GPU_flag:
            emb_u=self.UserPreference(torch.LongTensor(userid).cuda())
            emb_h = self.PoiPreference(Variable(torch.LongTensor(H)).cuda())
        else:
            emb_u = self.UserPreference(Variable(torch.LongTensor(userid)))
            emb_h = self.PoiPreference(Variable(torch.LongTensor(H)))
        m=emb_h.view(1,1,-1,self.emb_dimension)
        c1=self.C1(m)
        c2 = self.C2(F.pad(m,(0,0,1,1)))
        c3 = self.C3(F.pad(m,(0,0,3,3)))
        c4 = self.C4(F.pad(m, (0, 0, 7, 7)))
        uc=self.UC(emb_u.view(1,1,1,-1))
        hc=torch.cat((c1.view(self.channel,-1,self.emb_dimension),c2.view(self.channel,-1,self.emb_dimension),c3.view(self.channel,-1,self.emb_dimension),c4.view(self.channel,-1,self.emb_dimension)),1)
        uc=uc.view(self.channel,-1,self.emb_dimension)
        result=[]
        for i in range(self.channel):
            emb_u=uc[i]
            emb_h=hc[i]
            a=torch.mul(emb_u,emb_h)
            a=a.sum(1)
            a=a/math.sqrt(self.emb_dimension)
            a=F.softmax(a)
            a=a.view(1,-1)
            output=torch.mm(a,emb_h)
            output=output.view(1,-1)
            result.append(output)
        output=torch.cat(tuple(result),1)
        output=self.L1(output)
        output=self.L2(output)
        output=F.sigmoid(output)
        output=output.view(-1)
        return output