__author__ = 'yshi31'
from pandas import *
import os
from utils import *
import glob
# merge classified result with the raw data
folderlist = ['201403','201404','201405','201406','201407','201408','201409','201410']
ruleidlist = ['185']
column_need_to_keep = ['Idpost','bodypost','retweetCount','generatordname','kloutscore','matchingrulesvalues','languagevalue','favoritesCount','objpostedTime','objsummary','objlink','objid','actorpreferredusername','actorname','actorlinkshref','actorTimeZone','actorverified','actorstatusuesCount','actorsummary','actorlanguages','actorlink','actorfollowersCount','actorfavoritesCount','actorfriendsCount','actorPostedTime','actorid','actorObjType','postedTime','entitiesurlexpandedurl','entitiesmdaexpandedurl','objgeotype','objlocdname','locdname','locname','bodyoriginal','geocoordinates','geotype','gnipexpandedurl','gniploccountry','gniplocdname','gniplocgeocoordinates','gniploclocality','gniplocregion','gnipurl','Day','Month','Time','Year']
for i in range(len(folderlist)):
        monthly_data = DataFrame()
        path_out = os.path.join('H:/', 'Data','code','YARU','Ecig2014','results','monthly_data_for_Upenn', folderlist[i]+ '.csv')
        for j in range(len(ruleidlist)):
            print "checking rule",ruleidlist[j]
            path1 = os.path.join('H:/', 'Data','code','YARU','Ecig2014','classifier','classifer_relevant','result', folderlist[i]+ '_' + ruleidlist[j] + '_*_c.csv')
            path2 = os.path.join('H:/', 'Data','RawData_csv','GNIP','Twitterhistoricalpowertrack',folderlist[i]+'_Master','CSVRULES' , folderlist[i] + '_' + ruleidlist[j] + '_*.csv')
            possible_match1 = glob.glob(path1)
            possible_match2 = glob.glob(path2)
            for k in range(len(possible_match1)):
                merged_data = DataFrame()
                print "reading classified",possible_match1[k]
                try:
                    classified = read_csv(possible_match1[k],header= None,low_memory=False)
                    classified.columns = ['Idpost','bodypost','classifier']
                    classified = classified[['Idpost','classifier']]
                except Exception,e: print str(e)
                print "reading rawdata",possible_match1[k]
                try:
                    rawdata = read_csv(possible_match2[k],low_memory=False)
                    column_keep = []
                    column_not_have = []
                    for column in column_need_to_keep:
                        if column in rawdata.columns:
                            column_keep += [column]
                        else:
                            column_not_have += [column]
                    rawdata = rawdata[column_keep]
                    for column in column_not_have:
                        rawdata[column] = np.nan
                    merged_data = merge(classified,rawdata,on=['Idpost'])
                    merged_data = merged_data[merged_data.classifier == 1]
                    monthly_data = monthly_data.append(merged_data,ignore_index=True).drop_duplicates(subset = 'Idpost')
                except Exception,e: print str(e)
        monthly_data.to_csv(path_out,sep = ',',header=True,index=False)




