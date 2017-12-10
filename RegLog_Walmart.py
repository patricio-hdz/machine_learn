import pandas as pd
import numpy as np
import scipy as sp
from sklearn.linear_model import LogisticRegression
import time

start_time = time.time()

waltrain = pd.read_csv('C:/Users/lucky/Documents/Ciencia de Datos/Aprendizaje de Maquina/Proyecto Final ML/train/train.csv')
waltest = pd.read_csv('C:/Users/lucky/Documents/Ciencia de Datos/Aprendizaje de Maquina/Proyecto Final ML/test/test.csv')
waltrain = waltrain[waltrain.FinelineNumber.notnull()]
waltrain_part = waltrain[:]
waltest_part = waltest[:]

model = LogisticRegression()
x = waltrain_part[['Weekday', 'DepartmentDescription']]
y = waltrain_part[['TripType']]
x = pd.get_dummies(x)
z = waltest_part[['Weekday', 'DepartmentDescription']]
zend = pd.DataFrame({'Weekday': ['Sunday'],
'DepartmentDescription': ['HEALTH AND BEAUTY AIDS']},
index = [len(z)])
z = z.append(zend)
z = pd.get_dummies(z)

model.fit(x, y)
print ("The model coefficients are:")
print (model.coef_)
print ("The intercepts are:")
print (model.intercept_)
print ("model created after %f seconds" % (time.time() - start_time))

submission = model.predict_proba(z)
submissiondf = pd.DataFrame(submission)

submissiondf.drop(len(submissiondf)-1)

dex = waltest.iloc[:,0]
submurge = pd.concat([dex,submissiondf], axis = 1)
avgmurg = submurge.groupby(submurge.VisitNumber).mean()
avgmurg.reset_index(drop = True, inplace = True)
avgmurg.columns = ['VisitNumber', 'TripType_3','TripType_4','TripType_5','TripType_6','TripType_7',\
'TripType_8','TripType_9','TripType_12','TripType_14','TripType_15','TripType_18',\
'TripType_19','TripType_20','TripType_21','TripType_22','TripType_23','TripType_24',\
'TripType_25','TripType_26','TripType_27','TripType_28','TripType_29','TripType_30',\
'TripType_31','TripType_32','TripType_33','TripType_34','TripType_35','TripType_36',\
'TripType_37','TripType_38','TripType_39','TripType_40','TripType_41','TripType_42',\
'TripType_43','TripType_44','TripType_999']

avgmurg[['VisitNumber']] = avgmurg[['VisitNumber']].astype(int)
avgmurg.to_csv('C:/Users/lucky/Documents/Ciencia de Datos/Aprendizaje de Maquina/Proyecto Final ML/KaggleSub_04.csv', index = False)
print ("finished after %f seconds" % (time.time() - start_time))
