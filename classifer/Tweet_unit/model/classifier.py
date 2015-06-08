# coding=utf-8

from utils import loadLabels
from utils import getTweets
from utils import TestTweets

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm.classes import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model.logistic import LogisticRegression

from pandas import *


def buildMatrixTrain(tweet_list, tweet_label_list):
    corpus = []
    y = []
    for tweetid in tweet_list:
        document = ''.join(tweet_list[tweetid])
        try:
            corpus.append(document.encode('utf-8', 'ignore'))
        except:
            #print document
            continue
        if tweet_label_list[tweetid] == '1':
            y.append('+')
        else:
            y.append('-')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, y



def buildMatrixTrainAndTest(tweet_list, tweet_label_list, all_tweet):
    corpus = []
    y = []
    tweet_id_list = []
    for tweetid in tweet_list:
        document = ''.join(tweet_list[tweetid])
        try:
            corpus.append(document.decode('utf-8', 'ignore'))
            tweet_id_list.append(tweetid)
        except:
            print document
            continue
        if tweet_label_list[tweetid] == '1':
            y.append('+')
        else:
            y.append('-')
    for tweetid in all_tweet:
        document = ''.join(all_tweet[tweetid])
        try:
            corpus.append(document.decode('utf-8', 'ignore'))
            tweet_id_list.append(tweetid)
        except:
            #print document
            continue
    print len(corpus)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    #fnames = vectorizer.get_feature_names()
    #for x in fnames:
    #    print x
    return X, y, tweet_id_list


def splitTrainTest(X, y, start, end):
    import scipy.sparse as sp
    if end == X.shape[0]:
        X_train = X[:start]
    elif start == 0:
        X_train = X[end:]
    else:
        X_train = sp.vstack((X[:start], X[end:]), format='csr')
    y_train = y[:start] + y[end:]
    X_test = X[start:end]
    y_test = y[start:end]
    print 'dimension of training matrix : \t', X_train.shape, len(y_train)
    print 'dimension of testing  matrix : \t', X_test.shape, len(y_test)
    return X_train, y_train, X_test, y_test


def eval(y_test, y_pred):
    result = DataFrame({'y_pred' : y_pred,
                        'y_test' : y_test})
    crosstable = crosstab(result['y_pred'], result['y_test'])

    print crosstable

    acc = float(crosstable['+']['+']+crosstable['-']['-'])/len(y_test)
    recall = float(crosstable['+']['+'])/(crosstable['+']['+']+crosstable['-']['+'])
    prec = float(crosstable['+']['+'])/(crosstable['+']['+']+crosstable['+']['-'])
    F1 = 2 * prec * recall/( prec + recall)
    return acc, prec, recall, F1

def runModel(X, y, model_name):
    nFolders = 5
    accs = []
    precs = []
    recalls = []
    F1s = []

    n = X.shape[0]
    for exp in range(0, nFolders):
        print '\n\n============================================================================================\nexperiment' , exp
        ### 2.1 split training and testing data
        start = (int)((1-(exp+1) * 1.0/nFolders)*n)
        end = (int)((1-exp * 1.0/nFolders)*n)
        #print n, start, end
        X_train, y_train, X_test, y_test = splitTrainTest(X, y, start, end)
        print 'Running', model_name
        if model_name == 'SVM':
            ### 2.2 build classifier
            clf = LinearSVC(penalty="l1", dual=False, tol=1e-7)
            clf.fit(X_train, y_train)
            ### 2.3 predict
            y_pred = clf.predict(X_test)
        elif model_name == 'NaiveBayes':
            clf = GaussianNB()
            clf.fit(X_train.todense(), y_train)
            y_pred = clf.predict(X_test.todense())
        elif model_name == 'LogisticRegression':
            clf = LogisticRegression(C=1.0, penalty='l1', tol=0.01)
            clf.fit(X_train.toarray(), y_train)
            y_pred = clf.predict(X_test.toarray())
        else:
            raise Exception("The model name is incorrect!!!")
        ### 2.4 eval
        acc, prec, recall, F1 = eval(y_test, y_pred)
        print 'Acc = ', acc;
        print 'Precision =', prec;
        print 'Recall=', recall;
        print 'F1 =',  F1
        accs.append(acc)
        precs.append(prec)
        recalls.append(recall)
        F1s.append(F1)

    print '\n\n\n'
    print 'avg Acc = ', sum(accs)/len(accs)
    print 'avg Precision = ', sum(precs)/len(precs)
    print 'avg Recall = ', sum(recalls)/len(recalls)
    print 'avg F1 = ', sum(F1s)/len(F1s)
    return sum(accs)/len(accs), sum(precs)/len(precs),  sum(recalls)/len(recalls), sum(F1s)/len(F1s)

def plotResults(model_names, res_list):
    import numpy as np
    import matplotlib.pyplot as plt
    N = len(res_list[0])
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15       # the width of the bars
    fig, ax = plt.subplots(1,1, figsize=(20,10))
    rects1 = ax.bar(ind, res_list[0], width, color='red')
    rects2 = ax.bar(ind+width*1, res_list[1], width, color='green')
    rects3 = ax.bar(ind+width*2, res_list[2], width, color='blue')

    ax.set_xlabel('Evaluation Statistics')
    ax.set_ylabel('Value')
    ax.set_title('Model Comparison of LCC',fontsize=18)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(('Accuracy', 'Precision', 'Recall', 'F1') )
    ax.set_ylim(0.50,1)
    ax.legend( (rects1[0], rects2[0], rects3[0]), model_names, fancybox=True)
    plt.show()


def compareModel(user_tweets, user_vote_label):
    model_names = ('NaiveBayes', 'LogisticRegression', 'SVM')
    res_list = []

    X, y = buildMatrixTrain(user_tweets, user_vote_label)
    print 'dimension of entire data matrix : \t', X.shape, len(y)

    for model_name in model_names:
        A, P, R, F = runModel(X, y, model_name)
        res_list.append((A,P,R,F))

    plotResults(model_names, res_list)


def applyModel(tweet_list, tweet_label_list, all_tweet,filename):
    X, y, tweet_id_list= buildMatrixTrainAndTest(tweet_list, tweet_label_list, all_tweet)
    print X.shape, len(y)
    X_train = X[:len(y),:]
    y_train = y
    X_test =  X[len(y):,:]
    tweet_id_list_test = tweet_id_list[len(y):]
    print X_train.shape, X_test.shape
    clf = LinearSVC(penalty="l1", dual=False, tol=1e-7)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    with open(path1+filename+'_c.csv','w') as fp:
        for i, tweetid in enumerate(tweet_id_list_test):
            fp.write(tweetid +','+ y_pred[i]+'\n')


path1 = 'C:/Users/yshi31/Documents/LCC/classifer/classifer_relevant/result/'
path2 = 'H:/Data/RawData_csv/GNIP/Twitterhistoricalpowertrack/201410_LCC/'
#filelist = ['tw2014_10_01','tw2014_10_01_supp.json','tw2014_10_02','tw2014_10_02_supp.json','tw2014_10_03','tw2014_10_03_supp.json','tw2014_10_04','tw2014_10_04_supp.json','tw2014_10_05','tw2014_10_05_supp.json','tw2014_10_06','tw2014_10_06_supp.json','tw2014_10_07','tw2014_10_07_supp.json','tw2014_10_08','tw2014_10_09','tw2014_10_10','tw2014_10_11','tw2014_10_12','tw2014_10_13','tw2014_10_14','tw2014_10_15','tw2014_10_16','tw2014_10_17','tw2014_10_18','tw2014_10_19','tw2014_10_20','tw2014_10_21','tw2014_10_22','tw2014_10_23','tw2014_10_24','tw2014_10_25','tw2014_10_26','tw2014_10_27','tw2014_10_28','tw2014_10_29','tw2014_10_30','tw2014_10_31','tw_blunt_2014_10_01','tw_blunt_2014_10_02','tw_blunt_2014_10_03','tw_blunt_2014_10_04','tw_blunt_2014_10_05','tw_blunt_2014_10_06','tw_blunt_2014_10_07']
#filelist = ['tw_lccoriginal_2014_10_02','tw_lccoriginal_2014_10_03','tw_lccoriginal_2014_10_04','tw_lccoriginal_2014_10_05','tw_lccoriginal_2014_10_06','tw_lccoriginal_2014_10_07']
#filelist = ['tw2014_10_01','tw2014_10_02','tw2014_10_03','tw2014_10_04','tw2014_10_05','tw2014_10_06','tw2014_10_07','tw2014_10_08','tw2014_10_09','tw2014_10_10','tw2014_10_11','tw2014_10_12','tw2014_10_13','tw2014_10_14','tw2014_10_15','tw2014_10_16','tw2014_10_17','tw2014_10_18','tw2014_10_19','tw2014_10_20','tw2014_10_21','tw2014_10_22','tw2014_10_23','tw2014_10_24','tw2014_10_25','tw2014_10_26','tw2014_10_27','tw2014_10_28','tw2014_10_29','tw2014_10_30','tw2014_10_31']
tweet_label_list = loadLabels('sample_label')  #import tweets and according labels
tweet_list = getTweets('sample_label')
print(len(tweet_list))
#for fileId in filelist:
 #   all_tweet = TestTweets(path2,fileId)
  #  applyModel(tweet_list, tweet_label_list, all_tweet,fileId)
all_tweet = TestTweets('H:/Data/RawData_csv/GNIP/Twitterhistoricalpowertrack/201410_LCC/','tw_lccoriginal_2014_10_01')
print(len(all_tweet))
#applyModel(tweet_list, tweet_label_list, all_tweet,'tw_lccoriginal_2014_10_01')
#X, y, tweet_id_list= buildMatrixTrainAndTest(tweet_list, tweet_label_list, all_tweet)
#print X.shape, len(y)