__author__ = 'yshi31'
from utils import *

#random stratified sampling for machine classifier training

#1.1 randome sample for a cetain file
def random_sample(file,number,idloc,textloc,savename,seednumber):
    import csv
    import random
    random.seed(seednumber)
    tweet_list = {}
    with open (file, 'rb') as source:
        csvreader = csv.reader((line.replace('\0','') for line in source), delimiter=',', quotechar='"')
        for row in csvreader:
            tweet_list[row[textloc].lower()] = row[idloc]
        if len(tweet_list) > number:
           random_choice = random.sample(tweet_list.items(),number)
           print random_choice
           with open(RESULT_FOLDER+'/'+savename +'.csv','wb') as fp:
                  writer = csv.writer(fp, delimiter =",",lineterminator='\n',quoting=csv.QUOTE_MINIMAL)
                  for row in random_choice:
                      writer.writerow(row)
        else:
            print "not enough data to sample"


#1.2 stratified sampling based on each keyword
def stratified_random_sample(folderlist,monthlist,ruleidlist,samplenumberlist,path_root,textloc,idloc):
    import random
    for i in range(len(ruleidlist)):
        tweet_list = {}
        for j in range(len(monthlist)):
            path = path_root + folderlist[j] + '_Master/CSVRULES/' + monthlist[j]+ruleidlist[i]+'_1'
            with open (path, 'rb') as source:
                csvreader = csv.reader((line.replace('\0','') for line in source), delimiter=',', quotechar='"')
                for row in csvreader:
                    tweet_list[row[textloc].lower] = row[idloc]
        random_choice = random.sample(tweet_list.items(),samplenumberlist[i])
        with open(RESULT_FOLDER+'/'+ruleidlist[i] +'.csv','wb') as fp:
              writer = csv.writer(fp, delimiter =",",lineterminator='\n',quoting=csv.QUOTE_MINIMAL)
              for row in random_choice:
                  writer.writerow(row)


