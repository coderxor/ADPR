import csv
file1=open('./data/trainset.csv','r')
file2=open('./data/traindata12.csv','wb')
reader=csv.reader(file1)
writer=csv.writer(file2)
user={}
tao=12*3600
for data in reader:
    user.setdefault(data[0],[])
    line=[data[1],data[4]]
    user[data[0]].append(line)
for key in user.keys():
    for data1 in user[key]:
        line=[key,data1[1]]
        for data2 in user[key]:
            if data1[1]==data2[1]:
                continue
            if 0<=float(data1[0])-float(data2[0])<=tao:
                line.append(data2[1])
        writer.writerow(line)

file1.close()
file2.close()