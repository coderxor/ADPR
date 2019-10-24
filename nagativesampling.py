import csv
import random
import math

def random_pick(Prtable):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for key in Prtable.keys():
        cumulative_probability += Prtable[key]
        if x < cumulative_probability:
            return key
    return -1

neg_num=5
file1=open('./data/POI_info.csv','r')
POI_info={}
reader=csv.reader(file1)
for line in reader:
    POI_info.setdefault(line[0],line[2:4])
file1.close()
Cat2POI={}
POIcat={}
file2=open('./data/POIcat(id).csv','r')
reader=csv.reader(file2)
for line in reader:
    POIcat.setdefault(line[0],line[1:len(line)])
file2.close()
for key in POIcat.keys():
    for cat in POIcat[key]:
        Cat2POI.setdefault(cat,[])
        Cat2POI[cat].append(key)
file3=open('./data/traindata12.csv','r')
file4=open('./data/negativesample12.csv','wb')
reader=csv.reader(file3)
writer=csv.writer(file4)
print(POI_info)
for data in reader:
    Prtable={}
    count=0
    for cat in POIcat[data[1]]:
        for POI in Cat2POI[cat]:
            if POI==data[1]:
                continue
            if POI not in Prtable.keys():
                d=math.sqrt((float(POI_info[data[1]][0])-float(POI_info[POI][0]))**2+(float(POI_info[data[1]][1])-float(POI_info[POI][1]))**2)
                if d==0:
                    continue
                Prtable.setdefault(POI,1/d)
                count+=1/d
    for key in Prtable.keys():
        Pr=Prtable[key]/count
        Prtable[key]=Pr
    count = 0
    line = []
    while (count < neg_num):
        i = random_pick(Prtable)
        if int(i) < 0:
            print("error")
        if ((i != data[1])):
            line.append(i)
            count += 1
    writer.writerow(line)