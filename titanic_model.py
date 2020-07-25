#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
df_train=pd.read_csv("C:/Users/CNC/Desktop/train.csv")
df_test=pd.read_csv("C:/Users/CNC/Desktop/test.csv")
df_train.head()
df_test.head()
print(df_train.info())
df_train.describe()

#Visualisation
plt.figure(figsize=(10,10))
sns.barplot(df_train.Survived.value_counts().index,df_train.Survived.value_counts().values)
plt.xlabel("Survived")    
plt.title("Bar plot for survived")

plt.figure(figsize=(10,10))
sns.barplot(df_train.Sex.value_counts().index,df_train.Survived.value_counts().values)
plt.xlabel("Sex")
plt.ylabel("Survived")
plt.title("Bar plot for survived group by sex")

plt.figure(figsize=(10,10))
sns.barplot(df_train.Pclass.value_counts().index,df_train.Pclass.value_counts().values)
plt.xlabel("Pclass")
plt.title("Bar plot for Pclass")

plt.figure(figsize=(10,10))
sns.factorplot(x='Pclass',y='Survived',col='Sex',data=df_train,kind='bar',size=6)
plt.xlabel("Pclass")
plt.ylabel("Survived")
plt.title("Factor plot for Pclass vs survived group by sex")

plt.figure(figsize=(10,10))
sns.violinplot(x='Pclass',y='Age',hue='Survived',data=df_train,palette='Set2',split=True)
plt.xlabel("Pclass")
plt.ylabel("Survived")
plt.title("violin plot for Pclass vs Age group by Survived")

plt.figure(figsize=(10,10))
sns.factorplot(x='Embarked',y='Survived',data=df_train,kind='bar',size=6)
plt.xlabel("Embarked")
plt.ylabel("Survived")
plt.title("Factor plot for Embarked vs Survived")

plt.figure(figsize=(10,10))
sns.barplot(x='Embarked',y='Pclass',data=df_train)
plt.xlabel("Embarked")
plt.ylabel("Pclass")
plt.title("Bar plot for Embarked vs Pclass")

plt.figure(figsize=(10,10))
sns.factorplot(x='Survived',y='Pclass',col='Sex',data=df_train,kind='bar',size=6)
plt.xlabel("Pclass")
plt.ylabel("Survived")
plt.title("Factor plot for survived vs Pclass group by sex")

plt.figure(figsize=(10,10))
df_train.Fare[df_train.Pclass == 1].plot(kind='kde')    
df_train.Fare[df_train.Pclass == 2].plot(kind='kde')
df_train.Fare[df_train.Pclass == 3].plot(kind='kde')
plt.xlabel("Fare")    
plt.title("Fare Distribution within Pclass")
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')
#correlation matrix
corr=df_train.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, vmax=.6, linewidths=0.02,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features')


# data cleaning and feature engineering

bins=[0,16,32,48,64,80]
Age_band=pd.cut(df_train['Age'],bins)
bins=[0,20,40,60,80,100]
Fare_band=pd.cut(df_train['Fare'],bins)

encoder=LabelEncoder()
df_train['Sex']=encoder.fit_transform(df_train['Sex'])
df_train["Embarked"].fillna("S", inplace = True)
df_train['Embarked']=encoder.fit_transform(df_train['Embarked'])
df_train=df_train.drop(['PassengerId','Name','Age','Ticket','Fare','Cabin'],axis=1)
df_test['Sex']=encoder.fit_transform(df_test['Sex'])
df_test["Embarked"].fillna("S", inplace = True)
df_test['Embarked']=encoder.fit_transform(df_test['Embarked'])
df_test=df_test.drop(['PassengerId','Name','Age','Ticket','Fare','Cabin'],axis=1)

#spliting data set
dep_var=df_train.iloc[:,0:1]
ind_var=df_train.iloc[:,1:6]
x_train,x_test,y_train,y_test=train_test_split(ind_var,dep_var,train_size=0.75,random_state=0)

# Modeling

#KNN
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print("The accuracy for KNN model is %f" % accuracy_score(y_test,y_pred))
scores=cross_val_score(knn,x_train,y_train,cv=5)
print(scores)
print(scores.mean(),scores.std())
print("confusion matrix is")
print(confusion_matrix(y_test,y_pred))

# SVM
clf = svm.SVC()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy_score(y_test,y_pred)
print("The accuracy for SVM model is %f" % accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,x_train,y_train,cv=5)
print(scores)
print(scores.mean(),scores.std())
print("confusion matrix is")
print(confusion_matrix(y_test,y_pred))

# Decision tree
clfy = tree.DecisionTreeClassifier()
clfy.fit(x_train,y_train)
y_pred=clfy.predict(x_test)
accuracy_score(y_test,y_pred)
print("The accuracy for Decision tree model is %f" % accuracy_score(y_test,y_pred))
scores=cross_val_score(clfy,x_train,y_train,cv=5)
print(scores)
print(scores.mean(),scores.std())
print("confusion matrix is")
print(confusion_matrix(y_test,y_pred))

#Logistic regression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy_score(y_test,y_pred)
print("The accuracy for Logistic Regression model is %f" % accuracy_score(y_test,y_pred))
print("confusion matrix is")
print(confusion_matrix(y_test,y_pred))
scores=cross_val_score(model,x_train,y_train,cv=5)
print(scores)
print(scores.mean(),scores.std())


# Random Forest
forest=RandomForestClassifier()
forest.fit(x_train,y_train)
y_pred=forest.predict(x_test)
accuracy_score(y_test,y_pred)
print("The accuracy for Random Forest model is %f" % accuracy_score(y_test,y_pred))
scores=cross_val_score(forest,x_train,y_train,cv=5)
print(scores)
print(scores.mean(),scores.std())
print("confusion matrix is")
print(confusion_matrix(y_test,y_pred))

# Gaussian Naive Bays
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred=gnb.predict(x_test)
accuracy_score(y_test,y_pred)
print("The accuracy for Gaussian naive bays Regression model is %f" % accuracy_score(y_test,y_pred))
scores=cross_val_score(gnb,x_train,y_train,cv=5)
print(scores)
print(scores.mean(),scores.std())
print("confusion matrix is")
print(confusion_matrix(y_test,y_pred))

data={'Model':['KNN','SVM','Logistic Regression','Decision tree','Random forest','Naive bays'],'Mean':[0.7545,0.8054,0.7844,0.8024,0.7783,0.7815],'SD':[0.0561,0.0290,0.0178,0.0176,0.0316,0.0138]}
models_df=pd.DataFrame(data)
print(models_df)
output=[(0.79850746, 0.64925373, 0.78358209 ,0.7443609 , 0.79699248),(0.79850746 ,0.8358209  ,0.76865672 ,0.78195489 ,0.84210526),(0.81343284 ,0.7761194 , 0.79104478 ,0.78195489, 0.7593985),(0.79104478 ,0.81343284 ,0.7761194 , 0.80451128, 0.82706767),(0.79104478 ,0.79850746, 0.79104478, 0.7593985 , 0.72932331),(0.76865672, 0.7761194, 0.76865672 ,0.80451128, 0.78947368)]
model_names=['KNN','SVM','Logistic Regression','Decision tree','Random forest','Naive bays']
fig = plt.figure(figsize=(15,15))
fig.suptitle('Machine Learning Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(output)
ax.set_xticklabels(model_names)
plt.show()

