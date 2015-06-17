__author__ = 'yshi31'
from utils import *
import random
from skll import metrics

#agreement test between machine classifier and human label

def agreementtest(path1,path2):
    #1. import the labels
    from utils import loadLabels
    label_human = loadLabels(path1,0,2)
    label_machine = loadLabels(path2,0,2)
    #2. transfer them into the list
    y = []
    y_pred = []

    for key in label_human:
        y += [label_human[key]]
        y_pred += [label_machine[key]]
    print len(y),len(y_pred)
    #3. get the raw agreement
    from pandas import DataFrame
    from pandas import crosstab
    result = DataFrame({'y_pred' : y_pred,
                        'y_human' : y})
    crosstable = crosstab(result['y_pred'], result['y_human'])

    print crosstable

    acc = float(crosstable['1']['1']+crosstable['0']['0'])/len(y_pred)
    prec = float(crosstable['1']['1'])/(crosstable['1']['1']+crosstable['0']['1'])
    recall = float(crosstable['1']['1'])/(crosstable['1']['1']+crosstable['1']['0'])
    F1_hand = 2 * prec * recall/( prec + recall)

    #4. use the skll to get the kappa
    from skll import metrics
    kappa = metrics.kappa(y,y_pred)

    return crosstable,acc,recall,prec,F1_hand,kappa



#keyword precision based on human label
def precision(path,path_out):
    #1. import the data
    from pandas import read_csv
    from pandas import notnull
    human_label = read_csv(path +'.csv')
    #2.remove rows if two coders disagree
    human_label = human_label[human_label.EcigRelevance_coder1 == human_label.EcigRelevance_coder2]  #6960 lines remained
    #3.generate new column with code 1 0
    human_label['EcigRelevance_coder0No1Yes'] = (human_label.EcigRelevance_coder1 == 'Yes')
    #3.group by keyword and count the postive
    precision = human_label.groupby(['rule_index'])['EcigRelevance_coder0No1Yes'].mean()
    #4.output the result
    precision.to_csv(path_out,sep = ',',header=True)



    
#visulization
#trend of the total counts



