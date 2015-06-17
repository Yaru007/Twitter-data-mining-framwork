from ConfigParser import SafeConfigParser
import csv
import codecs
# coding=utf-8

parser = SafeConfigParser()
parser.read('config.ini')
DATA_FOLDER = parser.get('BASIC', 'DATA_FOLDER')
LABEL_FOLDER = parser.get('BASIC', 'LABEL_FOLDER')
RESULT_FOLDER = parser.get('BASIC', 'RESULT_FOLDER')


def loadLabels(path,idloc,labelloc):
    user_label = {}
    with open(path +'.csv') as fp:
    #with open(LABEL_FOLDER + '/' + file_name +'.csv') as fp:
        csvreader = csv.reader(fp, delimiter=',', quotechar='"')
        for row in csvreader:
            user_label[row[idloc]] = row[labelloc].lower()
    return user_label


def getTweets(path,idloc,textloc):
    tweet_list = {}
    with open(path +'.csv') as fp:
    #with open(LABEL_FOLDER + '/' + file_name +'.csv') as fp:
        csvreader = csv.reader(fp, delimiter=',', quotechar='"')
        for row in csvreader:
            tweet_list[row[idloc]] = row[textloc].lower()
    return tweet_list

def TestTweets(path,file_name):
    tweet_list = {}
    with open(path + file_name +'.csv','rb') as fp:
        csvreader = csv.reader((line.replace('\0','') for line in fp), delimiter=',', quotechar='"')
        line_count = 0
        for row in csvreader:
                tweet_list[row[0]] = row[1].lower()
                line_count += 1
    return tweet_list


#save the models
class Model:
    def __init__(self, model_name, classifier, feature_vectorizer):
        self.name = model_name;
        self.classifier = classifier;
        self.vectorizer = feature_vectorizer;

    def __str__(self):
        return self.name + ' classifier'

    def __repr__(self):
        return self.name + ' classifier'
