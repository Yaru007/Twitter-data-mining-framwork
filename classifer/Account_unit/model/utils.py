from ConfigParser import SafeConfigParser
from tweet import Tweet
import csv
import codecs
# coding=ACSII

parser = SafeConfigParser()
parser.read('config.ini')
DATA_FOLDER = parser.get('BASIC', 'DATA_FOLDER')
LABEL_FOLDER = parser.get('BASIC', 'LABEL_FOLDER')

def loadLabels(file_name1,file_name2):
    commercial_user_label = {}
    organic_user_label = {}
    relevant_list1 = {}
    with open(LABEL_FOLDER + '/' + file_name1 +'.csv') as fp:
        csvreader = csv.reader(fp, delimiter=',', quotechar='"')
        for row in csvreader:
            if row[1] == '0' or row[1] == '2':
                    commercial_user_label[row[0]] = '0'
                    relevant_list1[row[0]] = '0'
    with open(LABEL_FOLDER + '/' + file_name2 +'.csv') as fp:
        csvreader = csv.reader(fp, delimiter=',', quotechar='"')
        for row in csvreader:
            if row[3] == '0':
                if row[1] not in commercial_user_label:
                   commercial_user_label[row[1]] = row[3]
            elif row[3] == '1':
                if row[1] not in organic_user_label:
                   organic_user_label[row[1]] = row[3]
    return commercial_user_label,organic_user_label,relevant_list1

def getUserTweets(file_name1,file_name2,relevant_list):
        commercial_tweet_list = {}
        organic_tweet_list = {}
        with open(LABEL_FOLDER + '/' + file_name1 +'.csv') as fp:
            csvreader = csv.reader(fp, delimiter=',', quotechar='"')
            for row in csvreader:
                if row[1] in relevant_list:
                    if row[1] not in commercial_tweet_list:
                        commercial_tweet_list[row[1]] = [row[3]]
                    else:
                        commercial_tweet_list[row[1]].append(row[3])
        with open(LABEL_FOLDER + '/' + file_name2 +'.csv') as fp:
             csvreader = csv.reader(fp, delimiter=',', quotechar='"')
             for row in csvreader:
                if row[3] == '0':
                    if row[1] not in commercial_tweet_list:
                        commercial_tweet_list[row[1]] = [row[0]]
                    else:
                        commercial_tweet_list[row[1]].append(row[0])
                elif row[3] == '1':
                    if row[1] not in organic_tweet_list:
                        organic_tweet_list[row[1]] = [row[0]]
                    else:
                        organic_tweet_list[row[1]].append(row[0])
        return commercial_tweet_list,organic_tweet_list


def aug_data(dictionary_list):
    dictionary_list_aug = {}
    for key in dictionary_list:
        key_new = key + '1'
        dictionary_list_aug[key_new] = dictionary_list[key]
        dictionary_list_aug[key] = dictionary_list[key]
    return(dictionary_list_aug)


def getAllUserTweets(file_name):
        tweet_list = {}
        with open('H:/Data/RawData_csv/GNIP/Twitterhistoricalpowertrack/201410_LCC' + '/' + file_name +'.csv') as fp:
            csvreader = csv.DictReader((line.replace('\0','') for line in fp), delimiter=',', quotechar='"')
            #csvreader = csv.reader(fp, delimiter=',', quotechar='"').decode('utf-8', 'ignore')
            varlist = csvreader.fieldnames
            #print varlist
            for row in csvreader:
                 if 'actor_preferredUsername' in varlist:
                     if row['actor_preferredUsername'] not in tweet_list:
                         tweet_list[row['actor_preferredUsername']] = [row['body']]
                     else:
                         tweet_list[row['actor_preferredUsername']].append(row['body'])
                 else:
                     if row['actorpreferredusername'] not in tweet_list:
                         tweet_list[row['actorpreferredusername']] = [row['bodypost']]
                     else:
                         tweet_list[row['actorpreferredusername']].append(row['bodypost'])
        print "number of unique accounts=", len(tweet_list)
        return tweet_list

def loadSampleUserFeatures():
    csvfile = open(DATA_FOLDER + '/sample_user_features.csv');
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    header = reader.next()
    user_feature = {}
    for row in reader:
        user = row[0]
        user_feature[user] = row[1:]
        #print user, user_feature[user]
    return user_feature