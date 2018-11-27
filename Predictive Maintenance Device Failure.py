import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import ensemble, metrics
import gc
import csv
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

hdd=pd.read_csv('C://Project//Dataset//harddrive1.csv')
hdd.head()
# print(hdd.head())
np.int64(hdd['capacity_bytes'].at[5])
# print(np.int64(hdd['capacity_bytes'].at[5]))
# print(hdd['capacity_bytes'].at[5])
# print(hdd['capacity_bytes'].at[5] * 10 ** 10)
hdd.shape
# print(hdd.shape)
# number of hdd
hdd['serial_number'].value_counts().shape
# print(hdd['serial_number'].value_counts().shape)
# drop constant columns
hdd=hdd.loc[:, ~hdd.isnull().all()]
# number of different types of harddrives
hdd['model'].value_counts().shape
# print(hdd['model'].value_counts().shape)
# print(hdd.groupby('model')['failure'].sum().sort_values(ascending=False).iloc[:30])
hdd_st4=hdd.query('model == "ST4000DM000"')
del hdd
gc.collect()
# number of drives in the reduced data
hdd_st4['serial_number'].value_counts().shape
# out of the 35k drives there are 131 failures, so this is definitly an imbalanced dataset.
# note the output says 139 1 labeled but this is incorrect as 8 are duplicates. I drop them later
# because dropping at the begginning crashed the Kernel
hdd_st4['failure'].value_counts()
# print(hdd_st4['failure'].value_counts())
# more constant columns
hdd_st4['capacity_bytes'].value_counts()
# print(hdd_st4['capacity_bytes'].value_counts())
# drop them
hdd_st4=hdd_st4.loc[:, ~hdd_st4.isnull().all()]
hdd_st4.shape
# print(hdd_st4.shape)
# these have similar exponents as the size of harddrive.
# I am also pretty sure these variables are something like total read or total write.
# The scientific format is interprating the exponents as negative when they should likely be positive
# A fractional byte does not make sense.
hdd_st4.iloc[:5, 13:15]
# print(hdd_st4.iloc[:5,13:15])
# removed normalized values, and model, and capacity, since they are constants
hdd_st4=hdd_st4.select(lambda x: x[-10:] != 'normalized', axis=1)
hdd_st4=hdd_st4.drop(['model', 'capacity_bytes'], axis=1)
gc.collect()
# no null values left.
hdd_st4.isnull().any()


# turns number into a string, then extracts, base and exponent
def convert_large_number(large_num, min_exponent):
    str_num=str(large_num)

    base_end=str_num.find('e')
    base=np.float64((str_num[:base_end]))

    # if i remember correctly this is equivelent to dividing by a constant
    exponent=np.int64(str_num[base_end + 2:]) - (min_exponent - 1)
    return base * 10 ** exponent


# just fetches the exponent
def get_exp(large_num):
    str_num=str(large_num)
    base_end=str_num.find('e')

    exponent=np.int64(str_num[base_end + 2:])

    return exponent


# finds the minimum exponenet for a series
def min_exp(series_of_large_num):
    exps=series_of_large_num.apply(get_exp)
    return exps.min()


# scales a series down but subtracting the min observed exponent from exponent.
def scale_large_num_col(series, min_exponent):
    return series.apply(convert_large_number, min_exponent=min_exponent)


# smart_241_raw contains a single 0 which messes up my method of conversion
s241_mean=hdd_st4['smart_241_raw'].mean()
hdd_st4['smart_241_raw'].replace(0.0, s241_mean, inplace=True)

# transform data so it is a more managable size
# alternativly they could be stored as full length integers
for i in range(3, len(hdd_st4.columns)):
    if hdd_st4.iloc[0, i] < 10 ** -10 and hdd_st4.iloc[0, i] > 0:
        hdd_st4.iloc[:, i]=scale_large_num_col(hdd_st4.iloc[:, i], 308)

gc.collect()
hdd_st4.head()
# print(hdd_st4.head())

# Since we are trying to predict drive failure, we randomly select a set of drives.
# note that if there is some relationship between the drives, say a large group are in the same building. Then failure
# between drives won't be indepentent

hdd_st4.loc[:, 'date']=pd.to_datetime(hdd_st4.loc[:, 'date'])
hdd_st4['day_of_year']=hdd_st4['date'].dt.dayofyear

# hdd_st4.plot(kind='scatter', x='day_of_year', y='failure', title='Hard drive failures over time')
# plt.show()

# note this could be done earlier but it doesn't work on Kernels because of memory limitations
hdd_st4=hdd_st4.drop_duplicates()

# lets try to predict the probability of failure from data only on the day of failure
# it would be good to see how this probability relates to the probability of failure using previous days data only
hdd_group=hdd_st4.groupby('serial_number')
hdd_last_day=hdd_group.nth(-1)  # take the last row from each group
del hdd_st4
gc.collect()
print(gc.collect())

# the number of drives in the dataset
uniq_serial_num=pd.Series(hdd_last_day.index.unique())
uniq_serial_num.shape
#print(hdd_last_day)
hdd_last_day.to_csv('cleanData.csv', index=False)

# hold out 25% of data for testing
test_ids=uniq_serial_num.sample(frac=0.25)
train=hdd_last_day.query('index not in @test_ids')
test=hdd_last_day.query('index in @test_ids')
test['failure'].value_counts()
# print(test['failure'].value_counts())
train['failure'].value_counts()
# print(train['failure'].value_counts())
# close enough to stratified sampling for me.
131 / 4
train_labels=train['failure']
test_labels=test['failure']
train=train.drop('failure', axis=1)
test=test.drop('failure', axis=1)
train['day_of_year'].value_counts()

train=train.drop(['day_of_year', 'date'], axis=1)
test=test.drop(['day_of_year', 'date'], axis=1)
# remove more constant columns(anyone have a fast one liner for this?)
# could have done this earlier
for i in train.columns:
    if len(train.loc[:, i].unique()) == 1:
        train.drop(i, axis=1, inplace=True)
        test.drop(i, axis=1, inplace=True)
# print(train.head().columns)
rf=ensemble.RandomForestClassifier()
rf.fit(train, train_labels)
fitModel=rf.fit(train, train_labels)
preds=rf.predict_proba(test)

# rf=LogisticRegression()
# rf.fit(train, train_labels)
# fitModel=rf.fit(train, train_labels)
# preds=rf.predict_proba(test)

feature_cols=['smart_12_raw', 'smart_183_raw', 'smart_184_raw', 'smart_187_raw', 'smart_188_raw', 'smart_189_raw',
              'smart_190_raw', 'smart_192_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_198_raw',
              'smart_199_raw', 'smart_1_raw', 'smart_240_raw', 'smart_241_raw', 'smart_242_raw', 'smart_4_raw',
              'smart_5_raw', 'smart_7_raw', 'smart_9_raw']

# # SVM
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(train, train_labels)
# y_pred = svclassifier.predict(test)




# pd.DataFrame(test).to_csv('C:/Project/HardDriveTestModel/inputFile.csv', index=False)
# X_test_temp=pd.DataFrame(test)
#
# loadedModel=pickle.load(open('C:/Project/HardDriveTestModel/model.pkl', 'rb'))
# result_X_test=loadedModel.predict(X_test_temp)
# print("output", result_X_test)

# predictData = fitModel.predict(test)
# print("Predict TestModel",predictData)
# print('logloss', metrics.log_loss(y_true=test_labels, y_pred=preds[:,1]))
# print('roc_auc', metrics.roc_auc_score(y_true=test_labels, y_score=preds[:,1]))
rslt=rf.predict(test)
print(rslt)
# Accuracy
modelAccuracy=rf.score(test, test_labels)
print('Accuracy of Model is:-', modelAccuracy)

# X_test_temp['failure']=result_X_test
# print(X_test_temp)
# X_test_temp.to_csv('output.csv', index=False)

# SVM Accuracy
# modelAccuracy_SVM = svclassifier.score(test, test_labels)
# print('Accuracy of Model for SVM is:-', modelAccuracy_SVM)
# Confusion matrix
confusion_matrix=confusion_matrix(test_labels, rslt)
print("confusion_matrix", confusion_matrix)




# Validating
data=pd.read_csv("C:/Project/Dataset/checkdataset.csv", header=0)

y_pred_validate=rf.predict(data[feature_cols])
print(y_pred_validate)

data['check']=y_pred_validate
print("check",data)

data.to_csv('output.csv', index=False)

file=open('C:/Project/HardDriveTestModel/hard_drive_model.pkl', 'wb')

pickle.dump(fitModel, file)
file.close()

