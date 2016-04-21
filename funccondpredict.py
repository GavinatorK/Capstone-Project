import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sas7bdat import SAS7BDAT
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import random
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from precision_recall_auc import recall_precision_auc
from sklearn.metrics import auc, precision_recall_curve



def prauc(observed, predicted):
    precision, recall, thresholds = precision_recall_curve(observed, predicted)
    return(auc(recall, precision))



def conditionsymptoms(sclst, condlst, filepath):

    with SAS7BDAT(filepath) as f:
        intrv = f.to_data_frame()

    symcond=intrv[sclst]

    symcond['C3Q03']=symcond['C3Q03'].apply(lambda x: (-1) if x in [6,7] else x)
    symcond['C3Q03']=symcond['C3Q03'].apply(lambda x: 3 if np.isnan(x) else x)

    symcond.fillna(0, inplace=True)

    conditions=symcond[condlst]

    conditions.columns=['IDNUMXR', 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

    condition=pd.melt(conditions, id_vars='IDNUMXR')

    condition=condition[condition.value!=0]

    condlst.remove('IDNUMXR')
    symcond.drop(condlst, axis=1, inplace=True)

    return symcond, condition

def aucvals(predicted, observed):
    condnames=['ASTHMA','ATTENTION DEFICIT DISORDER','AUTISM','DOWN SYNDROME','MENTAL RETARDATION','EMOTIONAL PROBLEMS','DIABETES','CHILD USES INSULIN','HEART PROBLEM','BLOOD PROBLEMS','CYSTIC FIBROSIS','CEREBRAL PALSY','MUSCULAR DYSTROPHY','SEIZURE DISORDER','MIGRAINE OR FREQUENT HEADACHES','JOINT PROBLEMS','ALLERGIES','FOOD ALLERGIES']
    pred=np.array(predicted)
    obs=np.array(observed)
    auclst=[]
    for i in xrange(18):
        auclst.append((condnames[i], prauc(obs[:,i], pred[:,i])))
    return auclst

if __name__ == '__main__':

    filepath='2006ChildSpecialHealthCareNeeds/interview.sas7bdat'
    filepath09='2010ChildSpecialHealthCareNeeds/puf_cshcn_interview_unformat.sas7bdat'


    sclst=['IDNUMXR','AGE','C3Q02','C3Q03','S3Q01','S3Q01A','S3Q01B','S3Q02','S3Q02A','S3Q02B','S3Q03','S3Q04','S3Q05',
    'S3Q06','S3Q07','S3Q08','S3Q09','S3Q10','S3Q11','S3Q12','S3Q13','S3Q14','S3Q16','S3Q17','S3Q18','S3Q19','S3Q20','S3Q21','S3Q22','S3Q22A','S3Q23','S3Q25','S3Q26','S3Q27','S3Q28','S3Q29','S3Q30','S3Q32','S3Q31','S3Q31_A']

    sclst09=['IDNUMXR', 'AGE','C3Q02','C3Q03','C3Q21','C4Q05_9','C3Q21','C4Q05_10','C4Q05_10A','C4Q05_10C','C3Q23','C3Q24','C3Q25',
    'C3Q26','C3Q27','C3Q28','C3Q29','C3Q30','C3Q31','C3Q32','C3Q33','C3Q34','K2Q40B','K2Q31B','K2Q35B','K2Q50B','K2Q37B','K2Q34B','K2Q41B','K2Q41C','K2Q45B','K2Q46B','K2Q47B','K2Q48B','K2Q49B','K2Q42B','K2Q43B','K2Q51B','K2Q52B','K2Q52C' ]

    conditions=['IDNUMXR','S3Q16','S3Q17','S3Q18','S3Q19','S3Q20','S3Q21','S3Q22','S3Q22A','S3Q23','S3Q25','S3Q26','S3Q27','S3Q28','S3Q29','S3Q30','S3Q32','S3Q31','S3Q31_A']
    conditions09=['IDNUMXR','K2Q40B','K2Q31B','K2Q35B','K2Q50B','K2Q37B','K2Q34B','K2Q41B','K2Q41C','K2Q45B','K2Q46B','K2Q47B','K2Q48B','K2Q49B','K2Q42B','K2Q43B','K2Q51B','K2Q52B','K2Q52C']


    symcond05, condition05=conditionsymptoms(sclst, conditions,filepath )
    symcond09, condition09=conditionsymptoms(sclst09, conditions09,filepath09)
    symcond09.columns=['IDNUMXR','AGE','C3Q02','C3Q03','S3Q01','S3Q01A','S3Q01B','S3Q02','S3Q02A','S3Q02B','S3Q03','S3Q04','S3Q05',
'S3Q06','S3Q07','S3Q08','S3Q09','S3Q10','S3Q11','S3Q12','S3Q13','S3Q14']

    # symcond=symcond05.append(symcond09)
    # condition=condition05.append(condition09)

    dfXy05=symcond05.merge(condition05)
    dfXy09=symcond09.merge(condition09)
    X05=symcond05.drop('IDNUMXR', axis=1).values

    y05=dfXy05.variable.values.astype(int)
    y205=[[] for i in xrange(len(dfXy05['IDNUMXR'].unique()))]
    idlst05=dfXy05.IDNUMXR.unique()

    for i,each in enumerate(dfXy05['IDNUMXR'].unique()):

        y205[i]=list(dfXy05[dfXy05['IDNUMXR']==each]['variable'].values)

    x205=symcond05[symcond05['IDNUMXR'].isin(idlst05)].values

#2009

    X09=symcond09.drop('IDNUMXR', axis=1).values

    y09=dfXy09.variable.values.astype(int)
    y209=[[] for i in xrange(len(dfXy09['IDNUMXR'].unique()))]
    idlst09=dfXy09.IDNUMXR.unique()

    for i,each in enumerate(dfXy09['IDNUMXR'].unique()):

        y209[i]=list(dfXy09[dfXy09['IDNUMXR']==each]['variable'].values)

    x209=symcond09[symcond09['IDNUMXR'].isin(idlst09)].values


    y2=np.append(y205, y209, axis=0)
    x2=np.append(x205, x209, axis=0)

#n_estimators=500, min_samples_split=6, min_samples_leaf=4

    clf=RandomForestClassifier(n_estimators=300, min_samples_leaf=6, min_samples_split=10)

    x2=np.array(x2)
    y2=np.array(y2)

    y3=MultiLabelBinarizer().fit_transform(y2)

    clf_pipeline = OneVsRestClassifier(clf)



    # n_estimators=[300, 400]
    # min_samples_split=[8, 10]
    # min_samples_leaf=[4, 6, 10]
    # #best_params: {'estimator__min_samples_leaf': 4'estimator__min_samples_split': 10,'estimator__n_estimators': 300}
    # #class_weight=['balanced', 'balanced_subsample' ,None]
    #
    # param_grid = dict(estimator__n_estimators=n_estimators,
    #                   estimator__min_samples_split=min_samples_split,
    #                   estimator__min_samples_leaf=min_samples_leaf
    #                   )
    # grid = GridSearchCV(clf_pipeline, param_grid, scoring='f1_samples')
    # grid.fit(x2, y3)

    X_train, X_test, y_train, y_test = train_test_split(
         x2, y3, test_size=0.1, random_state=42)

#.fit(X_train, y_train)

    model=OneVsRestClassifier(clf)
    model.fit(X_train, y_train)



    # parameters = {'n_estimators':[200, 300, 400], 'min_samples_split':[4, 6, 8, 10], 'min_samples_leaf':[4,6,8]}
    #
    # gscv=GridSearchCV(model, param_grid, scoring="f1_samples")
    # gscv.fit(X_train,y_train)
    #



    predictions=model.predict(X_test)
    obs=y_test
    predprobs=model.predict_proba(X_test)



    sklearn.metrics.hamming_loss(y_test, predictions)
    sklearn.metrics.f1_score(y_test, predictions, average="samples")
    sklearn.metrics.precision_score(y_test, predictions)
    sklearn.metrics.recall_score(y_test, predictions)

    aucs=aucvals(predprobs, obs)
