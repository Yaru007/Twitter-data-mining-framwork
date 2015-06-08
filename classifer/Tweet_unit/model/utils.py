from ConfigParser import SafeConfigParser
import csv
import codecs
# coding=utf-8

parser = SafeConfigParser()
parser.read('config.ini')
DATA_FOLDER = parser.get('BASIC', 'DATA_FOLDER')
LABEL_FOLDER = parser.get('BASIC', 'LABEL_FOLDER')
RESULT_FOLDER = parser.get('BASIC', 'RESULT_FOLDER')


def loadLabels(file_name):
    user_label = {}
    with open(LABEL_FOLDER + '/' + file_name +'.csv') as fp:
        csvreader = csv.reader(fp, delimiter=',', quotechar='"')
        for row in csvreader:
            user_label[row[0]] = row[3].lower()
    return user_label


def getTweets(file_name):
    tweet_list = {}
    with open(LABEL_FOLDER + '/' + file_name +'.csv') as fp:
        csvreader = csv.reader(fp, delimiter=',', quotechar='"')
        for row in csvreader:
            tweet_list[row[0]] = row[1].lower()
    return tweet_list

def TestTweets(path,file_name):
    tweet_list = {}
    with open(path + file_name +'.csv','rb') as fp:
        csvreader = csv.reader((line.replace('\0','') for line in fp), delimiter=',', quotechar='"')
        for row in csvreader:
                tweet_list[row[0]] = row[1].lower()
    return tweet_list

# try:
#             for row in csvreader:
#                 tweet_list[row[0]] = row[1].strip().lower()
#         except csv.Error:
#             print csvreader.line_num