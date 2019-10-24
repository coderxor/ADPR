import csv
file1=open('./data/trainset.csv','r')
file2=open('./data/attentiondata.csv','wb')
reader=csv.reader(file1)
writer=csv.writer(file2)
user={}
tao=24*3600
for data in reader:
    user.setdefault(data[0],[])
    user[data[0]].append(data[4])
for key in user.keys():
    line=[key]
    line.extend(user[key])
    writer.writerow(line)
file1.close()
file2.close()