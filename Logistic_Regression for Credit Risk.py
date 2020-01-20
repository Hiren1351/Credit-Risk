# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:47:51 2020

@author: Hiru_Hunter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

loan_data = pd.read_csv('01Exercise1.csv')

loan_data.describe()
loan = loan_data.copy()

#Checking missing values

loan.isnull().sum(axis = 0)
sns.heatmap(loan.isnull())#Visualizing

#Fill the missing Value in Gender Column
from collections import Counter

Counter(loan['gender'])#Majority is male there is 112 female and 489 male 
loan['gender'].mode()

gender_null = loan[loan['gender'].isnull()].index.tolist()

loan['gender'].iloc[gender_null] = 'Male'

loan.isnull().sum(axis = 0)

#Fill the missing Value in ch Column

Counter(loan['ch'])#there is 475 1 and 89 0 so we replace with 1
loan['ch'].mode()

pd.crosstab(loan['ch'].isnull(),loan['status'])

#We can see that there is 37 missing in yes and 13 missing in no
ch_null = loan[(loan['status'] == 'Y') & (loan['ch'].isnull())].index.tolist()
loan['ch'].iloc[ch_null] = 1

ch_nulll = loan[(loan['status'] == 'N') & (loan['ch'].isnull())].index.tolist()
loan['ch'].iloc[ch_nulll] = 0

#Fill the missing Value in loanamt Column

loan['loanamt'].mean()

loanamt_null = loan[loan['loanamt'].isnull()].index.tolist() 
loan['loanamt'].iloc[loanamt_null] = 146 

#married column has only 3 missing value so we dropna

loan = loan.dropna()

#ch is a float but there is only two value 0 and 1 so we dont need to convert it

#Gender column is not impact much more for any bank to descide give loan or not so drop it

loan = loan.drop(['gender'],axis = 1)

#Create dummy variable

loan = pd.get_dummies(loan,drop_first = True)

#Scle the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

loan['income'] = scaler.fit_transform(loan[['income']])
loan['loanamt'] = scaler.fit_transform(loan[['loanamt']])

#Lets create X and Y variable or Features

X = loan.drop(['status_Y'],axis = 1)
Y = loan[['status_Y']]

#Split tha datainto Train and Test 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 1234,stratify = Y)


#Build the model

from sklearn.linear_model import LogisticRegression
Lg = LogisticRegression()
Lg.fit(X_train,Y_train)

Pred = Lg.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
cr1 = classification_report(Y_test,Pred)
cm1 = confusion_matrix(Y_test,Pred)

#out of 184 record model predict 36 incorrect in 34 is false positive that we want to descrese

#Create prediction Probability for checking threshold for sucess and failure

Y_prob = Lg.predict_proba(X_test)[:,1] #we check what %threshold for negative or 0

#lets set threshold 0.70% for above 80% predict positive or approve loan and below 70% reject for loan

#Threshold at 80%
Y_pred_new = []
threshold = 0.8

for i in range(0,len(Y_prob)):
    if Y_prob[i] > threshold:
        Y_pred_new.append(1)
    else:
        Y_pred_new.append(0)
        
cr2 = classification_report(Y_test,Y_pred_new)
cm2 = confusion_matrix(Y_test,Y_pred_new)

#We can see when we set threshold 80% than true positive rate descrese as well as false nagative also increase
#out of 184 record model predict 57 incorrect in 13 is false positive and 44 False Nagative that we want to descrese
#so lets change the threshold again from 80% to 78.5% and 60%

#Threshold at 78.5%
Y_pred_new1 = []
threshold = 0.785

for i in range(0,len(Y_prob)):
    if Y_prob[i] > threshold:
        Y_pred_new1.append(1)
    else:
        Y_pred_new1.append(0)
        
cr3 = classification_report(Y_test,Y_pred_new1)
cm3 = confusion_matrix(Y_test,Y_pred_new1)

#out of 184 record model predict 31 incorrect in 26 is false positive and 05F alse Nagative that we want to descrese


#Threshold at 60%
Y_pred_new2 = []
threshold = 0.60

for i in range(0,len(Y_prob)):
    if Y_prob[i] > threshold:
        Y_pred_new2.append(1)
    else:
        Y_pred_new2.append(0)
        
cr4 = classification_report(Y_test,Y_pred_new2)
cm4 = confusion_matrix(Y_test,Y_pred_new2)

#out of 184 record model predict 36 incorrect in 34 is false positive and 02 F alse Nagative that we want to descrese

#So,In Conclusion we can say that above 80% will aprove and below 60% reject
#Anything between 60-80 Hold

"""At 80% - so total 95 approve out of total 184
   At 60% - so total 26 Reject out of total 184
   Between 60-80% - Remaining is 63 on Hold or we can say Manual check"""
   
   
#Lets chech ROC AND AUC

from sklearn.metrics import roc_curve,roc_auc_score

fpr,tpr,threshold = roc_curve(Y_test,Y_prob)
auc = roc_auc_score(Y_test,Y_prob)#77% area under curve which is not bad

#Plot the ROC Curve

plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid()

"""We can see that in roc curve,in true positive rate thre is approx 58% 
value are False Positive in other word as per our confusion matrix out of 128 
True positive model is predict 34 as incorrect means false positive
actual is nagative but predict as positive and 28 predict correctly that 
actual is nagative and model also predict nagative true nagative so as per 
confusion matrix out of 128 there is 58 is false positive same as roc curve 60%"""

