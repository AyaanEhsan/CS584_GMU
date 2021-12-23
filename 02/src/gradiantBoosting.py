import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

#reading train dataset
train_data_raw = pd.read_csv("./credit_train.csv")

#Checking for missing values in the train dataset
train_data_raw.isnull().sum()

#dropping missing values in the train dataset
train_data_raw = train_data_raw.dropna(axis=0)

#converting categorical values of train data to indicator values
dummies_train = pd.get_dummies(train_data_raw[['F10', 'F11']])
train_data = pd.concat([train_data_raw.loc[:,'F1':'F9'], dummies_train], axis =1)

#Initializing the classifier
classifier = GradientBoostingClassifier(n_estimators= 1000,random_state=20)

#Fit the gradiant Boost classifier from the training dataset.
classifier_train=classifier.fit(train_data, train_data_raw['credit'])

#reading test dataset
test_data_raw = pd.read_csv("./credit_test.csv")

#converting categorical values of test data to indicator values
dummies_test = pd.get_dummies(test_data_raw[['F10', 'F11']])
test_data = pd.concat([test_data_raw.loc[:,'F1':'F9'], dummies_test], axis =1)

#predict the credit risk for test data
predictions=classifier.predict(test_data)

#writing the predictions to a file
pd.DataFrame(predictions).to_csv('predictions_GB.txt',index=False,header=False)

#initializing Kfold for cross validation
kFoldCV = KFold(n_splits=10, shuffle= True)

count =0
averageF1=0
#Cross validaion Logic
for train_split, test_split in kFoldCV.split(train_data_raw):
  credit_train, credit_test = pd.DataFrame(np.array(train_data_raw.iloc[:,:-1])[train_split]), pd.DataFrame(np.array(train_data_raw.iloc[:,:-1])[test_split])
  risk_train, risk_test = pd.DataFrame(np.array(train_data_raw.iloc[:, -1])[train_split]), pd.DataFrame(np.array(train_data_raw.iloc[:, -1])[test_split])
  dummies_train = pd.get_dummies(credit_train[[10,11]])
  train_data = pd.concat([credit_train.loc[:,1:9], dummies_train], axis =1)
  classifier.fit(train_data, risk_train.values.ravel())
  dummies_test = pd.get_dummies(credit_test[[10, 11]])
  test_data = pd.concat([credit_test.loc[:,1:9], dummies_test], axis =1)
  y_pred = classifier.predict(test_data)
  # Calculating the F1-Score for the split
  F1_score = f1_score(risk_test, y_pred)
  print("Split#",count+1, "\t","F1-Score",F1_score)
  count += 1
  averageF1=averageF1+F1_score
averageF1=averageF1/10
print("Average F1-Score on train data = ",averageF1)