import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

trainData = pd.read_csv('Data/train.dat',sep="\s+")
testData = pd.read_csv('Data/test.dat',sep="\s+")
movieActorsData = pd.read_csv('Data/movie_actors.dat',sep="\t")
movieDirectorsData = pd.read_csv('Data/movie_directors.dat',sep="\t")
movieGenreData = pd.read_csv('Data/movie_genres.dat',sep="\s+")
movieTagsData = pd.read_csv('Data/movie_tags.dat',sep="\s+")
userTaggedData = pd.read_csv('Data/user_taggedmovies.dat',sep="\s+")
tagsData = pd.read_csv('Data/tags.dat',sep="\t")

def getAvgGenreRating(trainData, movieGenreData, userID):
  avgGenreRating = dict()
  movieIDList = trainData[trainData['userID']==userID]['movieID'].values.tolist()
  movieRatingList = trainData[trainData['userID']==userID]['rating'].values.tolist()
  count = 0
  for movie, rating in zip(movieIDList, movieRatingList):
    movieGenre = movieGenreData[movieGenreData['movieID'] == movie]['genre'].values.tolist()
    for genre in movieGenre:
      avgGenreRatingKeys = avgGenreRating.keys()
      if genre not in avgGenreRatingKeys:
        avgGenreRating[genre] = [rating, 1]
      else:
        avgGenreRating[genre][0] += rating 
        avgGenreRating[genre][1] += 1
    count += 1
  for rating in avgGenreRating:
    sum = avgGenreRating[rating][0]
    values = avgGenreRating[rating][1]
    avgGenreRating[rating] = sum/values
  return avgGenreRating

def predict(trainData, testData, movieGenreData):
  predictions = []
  count = 0
  userGenre = None
  initialUserID = testData.values.tolist()[0]
  usersID = [i for i in set(trainData['userID'])]
  moviesID = [i for i in set(trainData['movieID'])]
  for testDataItem in testData.values.tolist():
    print(count)
    user = testDataItem[0]
    movie = testDataItem[1]
    if movie not in moviesID or user not in usersID: 
      predictions.append(1)
    else:
      if userGenre == None:
        userGenre = getAvgGenreRating(trainData, movieGenreData, user)
      if userGenre != None:
        if initialUserID != testDataItem[0]:
          userGenre = getAvgGenreRating(trainData, movieGenreData, user)
          initialUserID = testDataItem[0]
      movieGenres = movieGenreData[movieGenreData['movieID']==movie]['genre'].values.tolist()
      rating = 0
      for genres in movieGenres:
        if genres not in userGenre.keys():
          rating += 1.0
        else:
          rating += userGenre[genres]
      rating /= len(movieGenres)
      predictions.append(rating)
    count+=1
  return predictions

predictions = predict(trainData, testData, movieGenreData)

pd.DataFrame(predictions).to_csv("Predictions.txt",index=False,header=False)

#validation
uniqueUsers = trainData.drop_duplicates(subset=['userID'])
trainData = trainData[~trainData.isin(uniqueUsers)].dropna()
y_predicted = predict(trainData, uniqueUsers.iloc[:,:-1], movieGenreData)
RMSE = sqrt(mean_squared_error(uniqueUsers.iloc[:,-1], y_predicted))
print("Root mean squared error: "+str(RMSE))