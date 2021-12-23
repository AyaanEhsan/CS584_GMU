import pandas as pd
import numpy as np
import snowballstemmer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


#reading train dataset
train_data = pd.read_csv('train.txt', header = None, sep="#EOF", engine='python')
train_data = train_data[0].str.split('\t', 1, expand=True)
train_data=train_data.dropna()

trainData_reviews = train_data[1]
trainData_sentiments = train_data[0]

#function to pre process the reviews data
def preProcess(data):
    preProcessed_Data = []
    #initializing snowball stemmer (Porter2) to english language
    ss = snowballstemmer.stemmer('english')
    #getting stop_words in english as a set 
    stop_words = set(stopwords.words('english'))
    #removing HTML tags in the data
    data=data.apply(lambda x: re.sub('<.*?>', '', x))
    #preprocessing each review
    for i in range(len(data)):
      preProcessed_Review = " "
      #tokenizing the review into words
      words_list = word_tokenize(data.iloc[i])
      #removing punctuation
      word_list2 = [word for word in words_list if word not in ['`','~','!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','[','}','}','|','\\',':',';','"',"'",'<',',','>','.','?','/']]
      #removing numbers with words
      words_alphaOnly_list = [word for word in word_list2 if word.isalpha()]
      #as there are no adjective in 2-letters
      adjectives = [adj for adj in words_alphaOnly_list if len(adj)>2]
      #converting to lower case
      lowerCase_word_list = [w.lower() for w in adjectives]
      #applying snowball stemmer
      stemmedWords = [ss.stemWord(word) for word in lowerCase_word_list]
      #removing stop words
      preProcessed_Words = [word for word in stemmedWords if word not in stop_words]
      #final review after preprocessing and stemming
      preProcessed_Review = preProcessed_Review.join(preProcessed_Words)
      preProcessed_Data.append(preProcessed_Review)
    #returning final data after pre processing
    return preProcessed_Data

preProcessed_trainDataReviews = preProcess(trainData_reviews)

#splitting the train dataset into 10 fold by using kfold cross validation
kFoldCV = KFold(n_splits=10)
for train_split, test_split in kFoldCV.split(preProcessed_trainDataReviews):
  reviews_train, reviews_test = np.array(preProcessed_trainDataReviews)[train_split], np.array(preProcessed_trainDataReviews)[test_split]
  sentiments_train, sentiments_test = np.array(trainData_sentiments)[train_split], np.array(trainData_sentiments)[test_split]

#initializing tfidf vector with top 2000 features
feature_matrix = TfidfVectorizer(stop_words= 'english', min_df = 0.01, max_features = 2000)
#Sparse vector of train reviews using TFIDF 
train_tfidf_matrix = feature_matrix.fit_transform(reviews_train)
#Sparse vector of test reviews using TFIDF 
test_tfidf_matrix = feature_matrix.transform(reviews_test)

#function for K-nearest neighbors
def KNN(k, test_matrix, train_matrix, test_reviews, train_polarities):
    test_reviews_Polarity = []    
    for index in range(len(test_reviews)):
        #cosine similarity between test and train vectors
        cos_similarity = cosine_similarity(test_matrix[index], train_matrix).flatten()
        #getting the indices of k nearest neighbors and their sum 
        neighborIndices = cos_similarity.argsort()[:-k:-1]
        neighborsList = train_polarities[neighborIndices].tolist()
        neighborsList=[int(i) for i in neighborsList]
        sumOfNeighbors = sum(neighborsList)
        #predicting the review based on neighbors polarity sum
        test_reviews_Polarity.append(["+1" if sumOfNeighbors>0 else "-1"])  
    return pd.DataFrame(test_reviews_Polarity)

#Calculating the accuracy of the predictions based on train test split. 
Accuracy = []
highAcc=0
j=0
for i in range(1, 1001, 34):
  sentiments_prediction = KNN(i,test_tfidf_matrix, train_tfidf_matrix, reviews_test, sentiments_train)    
  acc = accuracy_score(sentiments_test.astype(int), sentiments_prediction.astype(int))
  Accuracy.append((i, acc))
  print(i, acc)
  if acc>highAcc:
    highAcc=acc
    j=i
print(j,highAcc)

X_axis = [dataPoint[0] for dataPoint in Accuracy]
Y_axis = [dataPoint[1] for dataPoint in Accuracy]
plt.plot(X_axis, Y_axis, color='b')
plt.xlabel('K-Value')
plt.ylabel('Accuracy')
plt.annotate('max K value',xy=(j,highAcc))
l="max K value = "+str(j)
plt.title(label=l)
plt.show()