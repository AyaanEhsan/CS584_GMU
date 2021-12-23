import pandas as pd
import snowballstemmer
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#reading train dataset
train_data = pd.read_csv('train.txt', header = None, sep="#EOF", engine='python')
train_data = train_data[0].str.split('\t', 1, expand=True)
train_data=train_data.dropna()

#reading test dataset
test_data = pd.read_csv('test.txt', header = None, sep="#EOF", engine='python')
test_data=test_data.drop(1,1)

#reviews from the train dataset
trainData_reviews = train_data[1]
#polarities from the train dataset
trainData_sentiments = train_data[0]
#reviews from the test dataset
testData_reviews = test_data[0]

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
      #as there are no adjective with 2-letters or less
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
#function for K-nearest neighbors
def KNN(k, test_matrix, train_matrix, test_reviews, train_polarities):
    test_reviews_Polarity = []    
    for index in range(len(test_reviews)):
        # cosine similarity between test and train vectors
        cos_similarity = cosine_similarity(test_matrix[index], train_matrix).flatten()
        #getting the indices of k nearest neighbors and calculating their sum
        neighborIndices = cos_similarity.argsort()[:-k:-1]
        neighborsList = train_polarities[neighborIndices].tolist()
        neighborsList=[int(i) for i in neighborsList]
        sumOfNeighbors = sum(neighborsList)
        #classify prediction review based on neighbor sum
        test_reviews_Polarity.append(["+1" if sumOfNeighbors>0 else "-1"])  
        print("review #"+str(index) +" is predicted")  
    return pd.DataFrame(test_reviews_Polarity)

#pre-processing the train data
preProcessed_trainDataReviews = preProcess(trainData_reviews)
#pre-processing the test data
preProcessed_testDataReviews = preProcess(testData_reviews)

#initializing tfidf vector with top 2000 features
feature_matrix = TfidfVectorizer(min_df = 0.01, max_features = 2000)
#Sparse vector of train reviews using TFIDF 
train_feature_matrix = feature_matrix.fit_transform(preProcessed_trainDataReviews)
#Sparse vector of test reviews using TFIDF 
test_feature_matrix = feature_matrix.transform(preProcessed_testDataReviews)

test_reviews = KNN(239, test_feature_matrix, train_feature_matrix, preProcessed_testDataReviews, trainData_sentiments)
#writing predicted polarities of test data to a file
test_reviews.to_csv('predictions.txt',index=False,header=False)