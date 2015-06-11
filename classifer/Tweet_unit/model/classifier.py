# coding=utf-8

from utils import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm.classes import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model.logistic import LogisticRegression
from model import *

from pandas import *
import pickle

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
    return X, y, vectorizer


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
    return X, y, tweet_id_list


def buildCorpusTest(all_tweet):
    corpus = []
    tweet_id_list = []
    for tweetid in all_tweet:
        document = ''.join(all_tweet[tweetid])
        try:
            corpus.append(document.decode('utf-8', 'ignore'))
            tweet_id_list.append(tweetid)
        except:
            print document
            continue
    print "number of tweets reading in is ", len(corpus)
    return corpus,tweet_id_list

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
    rects1 = ax.bar(ind, res_list[0], width, color='royalblue')
    rects2 = ax.bar(ind+width*1, res_list[1], width, color='lightcoral')

    ax.set_axis_bgcolor('lightgrey')
    ax.set_xlabel('Evaluation Statistics')
    ax.set_ylabel('Value')
    ax.set_title('Model Comparison of LCC',fontsize=18)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(('Accuracy', 'Precision', 'Recall', 'F1') )
    ax.set_ylim(0.75,1)
    ax.legend( (rects1[0], rects2[0]), model_names, fancybox=True)
    plt.show()


def compareModel(tweet_list, tweet_label_list):
    model_names = ('LogisticRegression', 'SVM')
    res_list = []
    result = []

    X, y = buildMatrixTrain(tweet_list, tweet_label_list)
    print 'dimension of entire data matrix : \t', X.shape, len(y)

    for model_name in model_names:
        A, P, R, F = runModel(X, y, model_name)
        res_list.append((A,P,R,F))

    plotResults(model_names, res_list)


def applyModel(tweet_list, tweet_label_list,all_tweet,model_name,filename):
    X, y, tweet_id_list= buildMatrixTrainAndTest(tweet_list, tweet_label_list, all_tweet)
    X_train = X[:len(y),:]
    y_train = y
    X_test =  X[len(y):,:]
    tweet_id_list_test = tweet_id_list[len(y):]
    #print "number of training tweets are ", X_train.shape, len(y_train)
    if model_name == 'SVM':
        clf = LinearSVC(penalty="l1", dual=False, tol=1e-7)
        clf.fit(X_train, y_train)
    elif model_name == 'NaiveBayes':
        clf = GaussianNB()
        clf.fit(X_train.todense(), y_train)
    elif model_name == 'LogisticRegression':
        clf = LogisticRegression(C=1.0, penalty='l1', tol=0.01)
        clf.fit(X_train.toarray(), y_train)
    else:
        raise Exception("The model name is incorrect!!!")

    y_pred = clf.predict(X_test)
    with open(RESULT_FOLDER+'/'+filename+'_c.csv','wb') as fp:
        writer = csv.writer(fp, delimiter =",",quoting=csv.QUOTE_MINIMAL)
        for i, tweetid in enumerate(tweet_id_list_test):
            writer.writerow([tweetid, all_tweet[tweetid], y_pred[i]])

def runandsaveModel(tweet_list, tweet_label_list,model_name):
    X, y, vectorizer= buildMatrixTrain(tweet_list, tweet_label_list)

    print "number of training tweets are ", X.shape, len(y)

    #trainning the model
    if model_name == 'SVM':
        clf = LinearSVC(penalty="l1", dual=False, tol=1e-7)
        clf.fit(X, y)
    elif model_name == 'NaiveBayes':
        clf = GaussianNB()
        clf.fit(X.todense(), y)
    elif model_name == 'LogisticRegression':
        clf = LogisticRegression(C=1.0, penalty='l1', tol=0.01)
        clf.fit(X.toarray(), y)
    else:
        raise Exception("The model name is incorrect!!!")

    #save the model
    model = Model(model_name, clf, vectorizer)
    with open(RESULT_FOLDER+"/"+model_name+"_model.m","wb") as pf:
        pickle.dump(model,pf)
    print model_name, "is saved at", RESULT_FOLDER+"/"+model_name+"_model.m"

def applyModel(model_name,test_corpus,tweet_id_list,filename):
    with open(RESULT_FOLDER+"/"+model_name+"_model.m","rb") as pf:
        model = pickle.load(pf)
    print "prediction results "
    featured_test_data = model.vectorizer.transform(test_corpus)
    y_pred = model.classifier.predict(featured_test_data)
    print y_pred
    with open(RESULT_FOLDER+'/'+filename+'_c.csv','wb') as fp:
        writer = csv.writer(fp, delimiter =",",quoting=csv.QUOTE_MINIMAL)
        for i, tweetid in enumerate(tweet_id_list):
            writer.writerow([tweetid, y_pred[i]])

if __name__ == '__main__':
