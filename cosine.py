
# coding: utf-8
import sys
import numpy as np
from pyspark import SparkContext
# In[30]:

#GetRatingFunction-Parse the Rating File
def GetRating(line):
    items = line.replace("\n", "").split(",")
    try:
        return int(items[0]), int(items[1]), float(items[2])
    except ValueError:
        pass


# In[31]:

#GetMovieFunction-Parse the Movie File
def GetMovie(line):
    items = line.replace("\n", "").split(",")
    try:
        return int(items[0]), items[1]
    except ValueError:
        pass


# In[32]:

#Calculate_MeanRating - This function caculates the average ratings of the users   
def Calculate_MeanRating(userRatingGroup):
    User_ID = userRatingGroup[0]
    Rating_Sum = 0.0
    Rating_Count = len(userRatingGroup[1])
    if Rating_Count == 0:
        return (User_ID, 0.0)
    for item in userRatingGroup[1]:
        Rating_Sum += item[1]
    return (User_ID, 1.0 * Rating_Sum / Rating_Count)


# In[33]:

#UserMovie_Construct - This constructs the ratings of the user as a list
def UserMovie_Construct(userRatingGroup):
    User_ID = userRatingGroup[0]
    movieList = [item[0] for item in userRatingGroup[1]]
    ratingList = [item[1] for item in userRatingGroup[1]]
    return (User_ID, (movieList, ratingList))


# In[34]:

#UserAvg_broadcast - this function broadcasts the user average rating RDD
def UserAvg_broadcast(sContext, UTrain_RDD):
    UserRatingAverage_List = UTrain_RDD.map(lambda x: Calculate_MeanRating(x)).collect()
    UserRatingAverage_Dict = {}
    for (user, avgscore) in UserRatingAverage_List:
        UserRatingAverage_Dict[user] = avgscore
    URatingAverage_BC = sContext.broadcast(UserRatingAverage_Dict)# broadcast
    return URatingAverage_BC


# In[35]:

#UserMovie_broadcast - this function broadcasts the User Movie List Dictionary 
def UserMovie_broadcast(sContext, UTrain_RDD):
    userMovieHistList = UTrain_RDD.map(lambda x: UserMovie_Construct(x)).collect()
    userMovieHistDict = {}
    for (user, mrlistTuple) in userMovieHistList:
        userMovieHistDict[user] = mrlistTuple
    uMHistBC = sContext.broadcast(userMovieHistDict)# broadcast
    return uMHistBC


# In[36]:

"""
Rating_construct - This function takes 2 users and their movies and ratings and returns the common movies amoung the users in
ascending order. 
Input is of the form - (userid,[movies,ratings]) 
output is of the form - ((user1,user2),[(ratingforcommonmovie_user1,ratingforcommonmovie_user2)])
"""
def Rating_construct(tup1, tup2):
    user1, user2 = tup1[0], tup2[0]
    mrlist1 = sorted(tup1[1])
    mrlist2 = sorted(tup2[1])
    ratepair = []
    index1, index2 = 0, 0
    while index1 < len(mrlist1) and index2 < len(mrlist2):
        if mrlist1[index1][0] < mrlist2[index2][0]:
            index1 += 1
        elif mrlist1[index1][0] == mrlist2[index2][0]:
            ratepair.append((mrlist1[index1][1], mrlist2[index2][1]))
            index1 += 1
            index2 += 1
        else:
            index2 += 1
    return ((user1, user2), ratepair)


# In[37]:

""" 
Cosine_Sim - This function calcualtes the cosine similarity for two user pair, the input for this function is the output of 
Rating_construct function and the output of this function is (user1,user2),(cosine similarity, number of common movies)
"""
import math
def Cosine_Sim(tup):
    dotproduct = 0.0
    sqsum1, sqsum2, cnt = 0.0, 0.0, 0
    for rpair in tup[1]:
        dotproduct += rpair[0] * rpair[1]
        sqsum1 += (rpair[0]) ** 2
        sqsum2 += (rpair[1]) ** 2
        cnt += 1
    denominator = math.sqrt(sqsum1) * math.sqrt(sqsum2)
    similarity = (dotproduct / denominator) if denominator else 0.0
    return (tup[0], (similarity, cnt))


# In[38]:

"""
User_GroupBy function takes in the input from the cosine similarity calculation fucntion and returns the single list
where each users are grouped by their ID. 
The output will be of the form (user1,(all the users, corresponding similarity, corresponding matching count)....)
"""
def User_GroupBy(record):
    return [(record[0][0], (record[0][1], record[1][0], record[1][1])), 
            (record[0][1], (record[0][0], record[1][0], record[1][1]))]


# In[39]:

"""
SimilarUser_pull - this function takes the userID, cosine similarity and the number of neighbors we wwant as input
and returns the corresponding number of neighbors.
"""
def SimilarUser_pull(user, records, numK = 200):
    llist = sorted(records, key=lambda x: x[1], reverse=True)
    llist = [x for x in llist if x[2] > 9]# filter out those whose cnt is small
    return (user, llist[:numK])


# In[40]:

"""
UserNeigh_broadcast - this function broadcast the userNeighborRDD
"""
def UserNeigh_broadcast(sContext, uNeighborRDD):
    userNeighborList = uNeighborRDD.collect()
    userNeighborDict = {}
    for user, simrecords in userNeighborList:
        userNeighborDict[user] = simrecords
    uNeighborBC = sContext.broadcast(userNeighborDict)
    return uNeighborBC


# In[41]:

"""
Error_calculation - This function takes the predicted RDD and actual RDD as input
This function returns the RMSE and MAE values

"""
def Error_calculation(predictedRDD, actualRDD):
    #initial transformation and joining the RDD
    predictedReformattedRDD = predictedRDD.map(lambda rec: ((rec[0], rec[1]), rec[2]))
    actualReformattedRDD = actualRDD.map(lambda rec: ((rec[0], rec[1]), rec[2]))
    joinedRDD = predictedReformattedRDD.join(actualReformattedRDD)
    #Calculating the Errors
    squaredErrorsRDD = joinedRDD.map(lambda x: (x[1][0] - x[1][1])*(x[1][0] - x[1][1]))
    totalSquareError = squaredErrorsRDD.reduce(lambda v1, v2: v1 + v2)
    numRatings = squaredErrorsRDD.count()
    return (math.sqrt(float(totalSquareError) / numRatings))


# In[42]:

""" 
Neighborhood_size - this function is used to invoke the previous error calculation fuunction and depending on the max number
of neighbors and step size, it iters and finds the corresponding error for for all those number of pairs.
"""
def Neighborhood_size(val4PredictRDD, validate_RDD, userNeighborDict, userMovieHistDict, UserRatingAverage_Dict, K_Range):
    errors = [0] * len(K_Range)
    err= 0
    for numK in K_Range:
        predictedRatingsRDD = val4PredictRDD.map(
            lambda x: User_prediction(x, userNeighborDict, userMovieHistDict, UserRatingAverage_Dict, numK)).cache()
        errors[err] = Error_calculation(predictedRatingsRDD, validate_RDD)
        err+= 1
    return errors


# In[43]:

""" 
User_prediction - this function predicts the rating
it takes the following as input,
the validationRDD, the neighbor dict whic has the user sim and corresponding count and id's
average rating of each user and the number of neighbors
"""
def User_prediction(tup, neighborDict, usermovHistDict, avgDict, topK):
   user, movie = tup[0], tup[1]
   avgrate = avgDict.get(user, 0.0)
   cnt = 0
   simsum = 0.0 # sum of similarity
   WeightedRating_Sum = 0.0
   neighbors = neighborDict.get(user, None)
   if neighbors:
       for record in neighbors:
           if cnt >= topK:
               break
           cnt += 1
           mrlistpair = usermovHistDict.get(record[0])
           if mrlistpair is None:
               continue
           index = -1
           try:
               index = mrlistpair[0].index(movie)
           except ValueError:# if error, then this neighbor hasn't rated the movie yet
               continue
           if index != -1:
               neighborAvg = avgDict.get(record[0], 0.0)
               simsum += abs(record[1])
               WeightedRating_Sum += (mrlistpair[1][index] - neighborAvg) * record[1]
   predRating = (avgrate + WeightedRating_Sum / simsum) if simsum else avgrate
   return (user, movie, predRating)
from collections import defaultdict


# In[44]:

"""
Final_recommend- this function takes the following inputs
ID of the user who we need recommendation for,
the RDD containg the userid and corresponding cosine similarity
the list of users adn every movie they have rated
maintain two dicts, one for similarity sum, one for weighted rating sum
for every neighbor of a user, get his rated items which hasn't been rated by current user
then for each movie, sum the weighted rating in the whole neighborhood 
and sum the similarity of users who rated the movie
iterate and sort
"""
def Final_recommend(user, neighbors, usermovHistDict, topK = 200, nRec = 5):
    simSumDict = defaultdict(float)# similarity sum
    weightedSumDict = defaultdict(float)# weighted rating sum
    movIDUserRated = usermovHistDict.get(user, [])
    for (neighbor, simScore, numCommonRating) in neighbors[:topK]:
        mrlistpair = usermovHistDict.get(neighbor)
        if mrlistpair:
            for index in range(0, len(mrlistpair[0])):
                movID = mrlistpair[0][index]
                simSumDict[movID] += simScore
                weightedSumDict[movID] += simScore * mrlistpair[1][index]# sim * rating
    candidates = [(mID, 1.0 * wsum / simSumDict[mID]) for (mID, wsum) in weightedSumDict.items()]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return (user, candidates[:nRec])


# In[45]:

#broadcastMovNameDict- This function broadcasts the movie RDD
def broadcastMovNameDict(sContext, movRDD):
    movieNameList = movRDD.collect()
    movieNameDict = {}
    for (movID, movName) in movieNameList:
        movieNameDict[movID] = movName
    mNameDictBC = sc.broadcast(movieNameDict)
    return mNameDictBC


# In[46]:

def genMovRecName(user, records, movNameDict):
    nlist = []
    for record in records:
        nlist.append(movNameDict[record[0]])
    return (user, nlist)


# In[47]:
#Spark program execution start
if __name__ == "__main__":
    
     #If input file is not provided, show error
    if len(sys.argv) !=3:
        print >> sys.stderr, "Usage: linreg <datafile> "
        exit(-1)
    #Initiatlize spark context
    sc = SparkContext(appName="Movie Recommendation Cosine Similarity")
#Reading the Data
ratings_raw_data = sc.textFile(sys.argv[1])
movies_raw_data = sc.textFile(sys.argv[2])


# In[48]:

#Removing the header from the rating data
ratingHeader = ratings_raw_data.first()
ratings_raw_data = ratings_raw_data.filter(lambda x: x != ratingHeader)
#Removing the header from the movies data
movieHeader = movies_raw_data.first()
movies_raw_data = movies_raw_data.filter(lambda x: x != movieHeader)


# In[49]:

# Moving the rating and movies data to ratingRDD and movies_RDD    
rating_RDD = ratings_raw_data.map(GetRating).cache()
movies_RDD = movies_raw_data.map(GetMovie).cache()


# In[50]:

ratings_Count = rating_RDD.count()
Movies_Count = movies_RDD.count()


# In[ ]:




# In[51]:

# Creating train and validate
Train_RDD, validate_RDD = rating_RDD.randomSplit([7,3])
ValPrediction_RDD = validate_RDD.map(lambda x: (x[0], x[1]))
TrainUserRating_RDD = Train_RDD.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().cache().mapValues(list)


# In[52]:

#TrainUserRating_RDD.take(5)


# In[53]:

#here we take the cartesian Product so as to get a Matrix so that later we can filter this matrix to get user pairs
UserRatingAverage_BC = UserAvg_broadcast(sc, TrainUserRating_RDD)
UserMovieList_BC = UserMovie_broadcast(sc, TrainUserRating_RDD)
cartesianUser_RDD = TrainUserRating_RDD.cartesian(TrainUserRating_RDD)


# In[54]:

#taking all the values below the diagonal so as to get user pairs
UserPairInitial_RDD = cartesianUser_RDD.filter(lambda x: x[0] < x[1])


# In[55]:

#invoking the cosine function and other RDD transformation functions
UserPair_RDD = UserPairInitial_RDD.map(
        lambda x: Rating_construct(x[0], x[1]))

User_Similarity_RDD = UserPair_RDD.map(lambda x: Cosine_Sim(x))

UserSimGroup_RDD = User_Similarity_RDD.flatMap(lambda x: User_GroupBy(x)).groupByKey()

UserNeighborhood_RDD = UserSimGroup_RDD.map(lambda x: SimilarUser_pull(x[0], x[1], 200))


# In[56]:

UserNeighborhood_BC = UserNeigh_broadcast(sc, UserNeighborhood_RDD)
Value_Error = [0]
#K_range is the starting number of neighbors and the ending number and the step size
K_Range = range(10, 210, 10)
err= 0
Value_Error[err] = Neighborhood_size(ValPrediction_RDD, validate_RDD, UserNeighborhood_BC.value, 
UserMovieList_BC.value, UserRatingAverage_BC.value, K_Range)
print(Value_Error)


# In[57]:

UserNeighborhood_RDD.map(lambda x: (x[1])).mapValues(list)
UserRecommendedMovie_RDD = UserNeighborhood_RDD.map(lambda x: Final_recommend(x[0], x[1], UserMovieList_BC.value))
MovieNameDict_BC = broadcastMovNameDict(sc, movies_RDD)
UserRecomendedMovie_RDD = UserRecommendedMovie_RDD.map(lambda x: genMovRecName(x[0], x[1], MovieNameDict_BC.value))


# In[58]:

#outputting Final Recommendations
print(UserRecomendedMovie_RDD.filter(lambda x: x[0]== 3).collect())

