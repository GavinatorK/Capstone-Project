import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sas7bdat import SAS7BDAT
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import random
from sklearn.grid_search import GridSearchCV
from precision_recall_auc import recall_precision_auc
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import make_scorer
from sklearn import svm
from scipy import stats

# Original DataFrame
'''unpack SAS files to dataframes'''

def unpack_sas(filepath):

    with SAS7BDAT(filepath) as f:
        intrv = f.to_data_frame()
    return intrv

'''modify dfs and get them ready for model'''

def modifydfs(intrv, collst, cond):

    dfsat=intrv[collst]
    dfsat["STATE"]=dfsat['STATE'].astype(int)


#dummyfying STATE
    dfsat=pd.concat([dfsat, pd.get_dummies(dfsat['STATE'], drop_first=True)], axis=1)
    dfsat.drop('STATE', axis=1, inplace=True)

#updating unknowns and replacing nans

    dfsat['C3Q03']=dfsat['C3Q03'].apply(lambda x: (-1) if x in [6,7] else x)
    dfsat['C3Q03']=dfsat['C3Q03'].apply(lambda x: 3 if np.isnan(x) else x)


    dfsat=dfsat.fillna(0)

    if 'C6Q0C' in collst:
        dfsat['C6Q0C']=dfsat['C6Q0C'].apply(lambda x: random.choice([1,2,3,4]) if x in [6,7]  else x)

        dfsat['C6Q0C']=dfsat['C6Q0C'].apply(lambda x: 0 if x in [1,2] else 1)

    dfsat['comorb']=dfsat[cond].sum(axis=1)
    return dfsat

'''to shuffle dataframes, not used at all'''
def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df


'''model to predict dissatisfaction'''
def model(dfsat, dfsat09):

    dfsat0=dfsat[dfsat['C6Q0C']==0].drop(['IDNUMR','IDNUMXR'], axis=1)
    dfsat1=dfsat[dfsat['C6Q0C']==1].drop(['IDNUMR','IDNUMXR'], axis=1)

    y0=dfsat0.pop('C6Q0C').values
    y1=dfsat1.pop('C6Q0C').values

    #lstf=dfsat.drop(['IDNUMR','IDNUMXR'], axis=1).columns.tolist()
    Y=dfsat.pop('C6Q0C').values
    X=dfsat.drop(['IDNUMR','IDNUMXR'], axis=1)

    Xo_train, Xo_test, yo_train, yo_test = train_test_split(
         X, Y, test_size=0.99)




    lstf=dfsat09.drop(['IDNUMR','IDNUMXR'], axis=1).columns.tolist()
    ids=dfsat09.IDNUMXR
    X2=dfsat09.drop(['IDNUMR','IDNUMXR'], axis=1).values

    '''
    Model below here
    '''

    clflst=[None]*10
    pred=[None]*10
    clflst2=[None]*10
    predsvm=[None]*10
    for i in xrange(10):
        x1=dfsat1.values
        #dfsat0s=shuffle(dfsat0)
        x0=dfsat0.sample(n=10000).values
        x=np.concatenate((x1, x0), axis=0)
        y=np.concatenate((y1,np.zeros(10000)), axis=0)
    #    X_train, X_test, y_train, y_test = train_test_split(
    #         x, y, test_size=0.001)
        clflst[i]=RandomForestClassifier(n_estimators=400, min_samples_split=2, min_samples_leaf=2)
        # clflst2[i]=svm.SVC(probability=True)
        # clflst2[i].fit(X_train, y_train)
        clflst[i].fit(x, y)
        # print clflst[i].score(Xo_test, yo_test)
        # print clflst2[i].score(Xo_test, yo_test)
        pred[i]=clflst[i].predict(X2)
        #predsvm[i]=clflst2[i].predict(Xo_test)


    preds=np.reshape(pred, (10, len(pred[0])))
    preds2=[stats.mode(preds[:,i])[0][0] for i in xrange(len(preds[0]))]
    return preds2





if __name__ == '__main__':
    intrv=unpack_sas('2006ChildSpecialHealthCareNeeds/interview.sas7bdat')
    intrv09=unpack_sas('2010ChildSpecialHealthCareNeeds/puf_cshcn_interview_unformat.sas7bdat')
    collst06=['IDNUMR', 'IDNUMXR', 'STATE','AGE','MSASTATR','FAMSTRUCT','C8Q01_A', 'UNINS', 'C3Q02','C3Q03','S3Q01','S3Q01A','S3Q01B','S3Q02','S3Q02A','S3Q02B','S3Q03','S3Q04','S3Q05',
    'S3Q06','S3Q07','S3Q08','S3Q09','S3Q10','S3Q11','S3Q12','S3Q13','S3Q14','S3Q16','S3Q17','S3Q18','S3Q19','S3Q20','S3Q21','S3Q22','S3Q22A','S3Q23','S3Q25','S3Q26','S3Q27','S3Q28','S3Q29','S3Q30','S3Q32','S3Q31','S3Q31_A','C6Q0C',]
    cond06=['S3Q16','S3Q17','S3Q18','S3Q19','S3Q20','S3Q21','S3Q22','S3Q22A','S3Q23','S3Q25','S3Q26','S3Q27','S3Q28','S3Q29','S3Q30','S3Q32','S3Q31','S3Q31_A']
    collst10=['IDNUMR', 'IDNUMXR', 'STATE','AGE','MSASTATR','FAMSTRUCT','C8Q01_A', 'UNINS', 'C3Q02','C3Q03','C3Q21','C4Q05_9','C3Q21','C4Q05_10','C4Q05_10A','C4Q05_10C','C3Q23','C3Q24','C3Q25',
    'C3Q26','C3Q27','C3Q28','C3Q29','C3Q30','C3Q31','C3Q32','C3Q33','C3Q34','K2Q40B','K2Q31B','K2Q35B','K2Q50B','K2Q37B','K2Q34B','K2Q41B','K2Q41C','K2Q45B','K2Q46B','K2Q47B','K2Q48B','K2Q49B','K2Q42B','K2Q43B','K2Q51B','K2Q52B','K2Q52C']
    cond10=['K2Q40B','K2Q31B','K2Q35B','K2Q50B','K2Q37B','K2Q34B','K2Q41B','K2Q41C','K2Q45B','K2Q46B','K2Q47B','K2Q48B','K2Q49B','K2Q42B','K2Q43B','K2Q51B','K2Q52B','K2Q52C']

    dfsat=modifydfs(intrv, collst06, cond06)
    dfsat09=modifydfs(intrv09, collst10, cond10)

    y2=model(dfsat, dfsat09)
    dfsat10=pd.DataFrame()
    dfsat10['C6Q0C']=y2
    dfsat10.to_csv('predsat2009.csv')
