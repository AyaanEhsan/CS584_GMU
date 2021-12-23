Files:
1. hw1.py   
    - the function KNN in this file has the implementation of the K-Nearest Neighbors Algorithm
    - the function preProcess in this file has all the steps I followed for pre-processing the given dataset, stemming
    - I have used k as 239 in the KNN to get an accuracy of 80. I obtained this k values after running the cross validation
    - use the command python hw1.py to run this file and this generates the Predictions.txt file which will have the polarities of all the reviews in the test data given
2. hw1_cv.py 
    - this file has the KFold cross validation implementation to test the accuracy of the polarities with the help of train data
    - use python hw1_cv.py to run this file and get the K value that has high accuracy in prediction
3. test.txt
    - it has the given test data
4. train.txt
    - it has the given train data
5. predictions.txt
    - it has the predicted polarities of all the reviews in the test data
