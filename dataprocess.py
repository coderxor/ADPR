import csv
file1=open('./data/CheckIn_6.csv','r')
file2=open('./data/trainset.csv','wb')
file3=open('./data/tuningset.csv','wb')
file4=open('./data/testset.csv','wb')
reader=csv.reader(file1)
writer1=csv.writer(file2)
writer2=csv.writer(file3)
writer3=csv.writer(file4)
NumofRecords=134347
train=0.5
tunning=0.0
for i,data in enumerate(reader):
    if i<=int(NumofRecords*train):
        writer1.writerow(data)
    elif i<=int(NumofRecords*(train+tunning)):
        writer2.writerow(data)
    else:
        writer3.writerow(data)