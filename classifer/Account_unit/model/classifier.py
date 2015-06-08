from tweet import Tweet
from utils import *
#from utils import DATA_FOLDER
#from utils import loadLabels_big
#from utils import loadLabels_append
#from utils import getUserTweets_big
#from utils import getUserTweets_append
#from utils import loadSampleUserFeatures
#from utils import loadAllUserTweets

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm.classes import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import svm
from prettytable import PrettyTable
from pandas import *


def buildMatrixTrain(user_tweets, user_vote_label):
    corpus = []
    y = []
    for user in user_tweets:
        document = ''.join(user_tweets[user])
        try:
            corpus.append(document.encode('utf-8', 'ignore'))
        except:
            #print document
            continue
        if user_vote_label[user] == '0':
            y.append('+')
        else:
            y.append('-')        
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    #fnames = vectorizer.get_feature_names()
    #for x in fnames:
    #    print x
    return X, y


def buildMatrixTrainAndTest(user_tweets, user_vote_label, allusers_tweets):
    corpus = []
    y = []
    user_id_list = []
    for user in user_tweets:
        document = ''.join(user_tweets[user])
        try:
            corpus.append(document.decode('utf-8', 'ignore'))
            user_id_list.append(user)
        except:
            #print document
            continue
        
        if user_vote_label[user] == '0':
            y.append('+')
        else:
            y.append('-')
    for user in allusers_tweets:
        document = ''.join(allusers_tweets[user])
        try:
            corpus.append(document.decode('utf-8', 'ignore'))
            user_id_list.append(user)
        except:
            #print document
            continue
            
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    fnames = vectorizer.get_feature_names()
    #for x in fnames:
    #    print x
    return X, y,user_id_list

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
        if model_name == 'SVM_new':
            ### 2.2 build classifier
            clf = svm.SVC(C = 1.0, gamma = 1.0, class_weight = 'auto')
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
    width = 0.2       # the width of the bars
    fig, ax = plt.subplots(1,1, figsize=(20,10))
    rects1 = ax.bar(ind, res_list[0], width, color='r')
    rects2 = ax.bar(ind+width*1, res_list[1], width, color='y')
    rects3 = ax.bar(ind+width*2, res_list[2], width, color='b')
    
    ax.set_xlabel('Evaluation Metrics')
    ax.set_ylabel('Model Performance')
    ax.set_title('Model Comparision of Twitter E-cigeratte Spammers Detection',fontsize=18)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(('Accuracy', 'Precision', 'Recall', 'F1')  )
    ax.set_ylim(0.8,1)
    ax.legend( (rects1[0], rects2[0], rects3[0]), model_names, fancybox=True)
    plt.show()
    

def compareModel(user_tweets, user_vote_label):
    model_names = ('LogisticRegression', 'SVM')
    res_list = []
    
    X, y = buildMatrixTrain(user_tweets, user_vote_label)
    print 'dimension of entire data matrix : \t', X.shape, len(y)
    print
    
    for model_name in model_names:
        A, P, R, F = runModel(X, y, model_name)
        res_list.append((A,P,R,F))
    
    plotResults(model_names, res_list)
    
def applyModel(user_tweets, user_vote_label, alluser_tweets,filename):
    X, y, user_id_list = buildMatrixTrainAndTest(user_tweets, user_vote_label, alluser_tweets)
    #print X.shape, len(y), len(user_id_list)
    X_train = X[:len(y),:]
    y_train = y
    X_test =  X[len(y):,:]
    user_id_list_test = user_id_list[len(y):]
    #print X_train.shape, len(y_train), X_test.shape

    #print X_train.shape, X_test.shape
    clf = svm.SVC(C = 1.0, gamma = 1.0, class_weight = 'auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print "length of y_pred", len(y_pred)
    
    with open(DATA_FOLDER+'/labels/'+filename+'.csv','w') as fp:
        for i, user in enumerate(user_id_list_test):
            fp.write(user+','+y_pred[i]+'\n')
        

#if __name__ == '__main__':
#user_label_random, unrelevant_user = loadLabels_big('sample_user_labeled_resultsLCC')
#print len(user_label_random), len(unrelevant_user)
commercial_user_label,organic_user_label,relevant_list1= loadLabels('sample_user_labeled_resultsLCC','LCC_sample_5_23AAAcMar_full')
#print len(commercial_user_label), len(organic_user_label), len(relevant_list1)


commercial_tweet_list,organic_tweet_list =  getUserTweets('sample_user_tweets','LCC_sample_5_23AAAcMar_full',relevant_list1)
#print len(commercial_tweet_list), len(organic_tweet_list)
#sampleuser_tweets = getUserTweets('sample_user_tweets', 'LCC_sample_5_23AAAcMar_full',relevant_list)
#print len(sampleuser_tweets)

#augment commercial users
commercial_user_label_aug = aug_data(commercial_user_label)
commercial_tweet_list_aug = aug_data(commercial_tweet_list)
#print len(commercial_user_label_aug), len(commercial_tweet_list_aug)


user_label = organic_user_label.copy()
user_label.update(commercial_user_label_aug)

sampleuser_tweets = organic_tweet_list.copy()
sampleuser_tweets.update(commercial_tweet_list_aug)

#print len(sampleuser_tweets), len(user_label)
#print len(sampleuser_tweets), len(user_label)
#X, y = buildMatrixTrain(sampleuser_tweets, user_label)
#a,b,c,d = runModel(X, y, 'SVM_new')

def trainAndSaveOneModel(X, y, model_name, model_save_path):
    n = X.shape[0]
    clf = svm.SVC(C = 1.0, gamma = 1.0, class_weight = 'auto')
    clf.fit(X, y)
    import pickle
    print "start saving the model ..."
    pickle.dump(clf, open(model_save_path+"/"+model_name+"_model.m","wb"))
    print model_name +" model has successfully saved!\n"
#models = trainAndSaveOneModel(X,y,'SVM_new',"C:/Users/yshi31/Documents/LCC/classifer/classifier_account/model/trained_model")

#alluser_tweets = getAllUserTweets('tw2014_10_01')
#print len(alluser_tweets)
#applyModel(sampleuser_tweets, user_label, alluser_tweets,'tw2014_10_06')

filelist = ['tw_lccoriginal_2014_10_01','tw_lccoriginal_2014_10_02','tw_lccoriginal_2014_10_03','tw_lccoriginal_2014_10_04','tw_lccoriginal_2014_10_05','tw_lccoriginal_2014_10_06','tw_lccoriginal_2014_10_07','tw2014_10_01','tw2014_10_01_supp.json','tw2014_10_02','tw2014_10_02_supp.json','tw2014_10_03','tw2014_10_03_supp.json','tw2014_10_04','tw2014_10_04_supp.json','tw2014_10_05','tw2014_10_05_supp.json','tw2014_10_06','tw2014_10_06_supp.json','tw2014_10_07','tw2014_10_07_supp.json','tw2014_10_08','tw2014_10_09','tw2014_10_10','tw2014_10_11','tw2014_10_12','tw2014_10_13','tw2014_10_14','tw2014_10_15','tw2014_10_16','tw2014_10_17','tw2014_10_18','tw2014_10_19','tw2014_10_20','tw2014_10_21','tw2014_10_22','tw2014_10_23','tw2014_10_24','tw2014_10_25','tw2014_10_26','tw2014_10_27','tw2014_10_28','tw2014_10_29','tw2014_10_30','tw2014_10_31','tw_blunt_2014_10_01','tw_blunt_2014_10_02','tw_blunt_2014_10_03','tw_blunt_2014_10_04','tw_blunt_2014_10_05','tw_blunt_2014_10_06','tw_blunt_2014_10_07']
for fileId in filelist:
     print "reading", fileId
     alluser_tweets = getAllUserTweets(fileId)
     applyModel(sampleuser_tweets, user_label, alluser_tweets,fileId)

#sampleuser_tweets_random = getUserTweets_big('sample_user_tweets', unrelevant_user)
#print len(sampleuser_tweets_random)
#sampleuser_tweets = getUserTweets_append('LCC_sample_5_23AAAcMar_full',sampleuser_tweets_random)
#print len(sampleuser_tweets)


    #print
#compareModel(sampleuser_tweets, user_label)
# model_names = ('LogisticRegression', 'SVM')




	
    #alltweets = loadAllUserTweets()
    #alluser_tweets = getUserTweets(alltweets)
    #print len(alluser_tweets)
    #print
    
    #applyModel(sampleuser_tweets, user_vote_label, alluser_tweets)