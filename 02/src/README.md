Files:

1. featureCorrelation.py
   - use the command *python featureCorrelationhw1.py* to run this file and this generates the correlation heatmap between all the available features in the dataset
2. KNN.py
   - this file has the `KNeighborsClassifier` and KFold cross validation implementation to get the F1-Score on the train data
   - use *python KNN.py* to run this file and it generates predictions_KNN.txt which has the predicted credit risks for the test data
3. *NaiveBayes.py*
   - this file has the `GaussianNB` and KFold cross validation implementation to get the F1-Score on the train data
   - use *python NaiveBayes.py* to run this file and it generates predictions_NB.txt which has the predicted credit risks for the test data
4. SVM.py
   * this file has the `svm.LinearSVC` and KFold cross validation implementation to get the F1-Score on the train data
   * use *python SVM.py* to run this file and it generates predictions_SVM.txt which has the predicted credit risks for the test data
5. *RandomForest.py*
   * this file has the `ExtraTreesClassifier` and KFold cross validation implementation to get the F1-Score on the train data
   * use *python RandomForest.py* to run this file and it generates predictions_RF.txt which has the predicted credit risks for the test data
6. *DecisionTree.py*
   * this file has the `DecisionTreeClassifier` and KFold cross validation implementation to get the F1-Score on the train data
   * use *python DecisionTree.py* to run this file and it generates predictions_DT.txt which has the predicted credit risks for the test data
