'''
Cloud Computing Project ALS implementation
Group 21
Team: Satish Kumar, Abhijit Nair, Kshitij Khurana
'''

from pyspark import SparkContext
import numpy as np
import json
from numpy.random import rand
from numpy import matrix
import sys

#For calculating RMSE    
def get_rms_error(rating_mat, movie_mat, user_mat):
    movUserTrans = movie_mat * user_mat.T
    val_differ = rating_mat - movUserTrans
    val_differ_sq = (np.power(val_differ, 2)) / (movie_mat_row * user_mat_row)
    return np.sqrt(np.sum(val_differ_sq))

#For Fixing the movie matrix
def fix_movie_mat(x):
    u_rate_Trans = broad_rating_mat.value.T
    mmTrans = broad_movie.value.T * broad_movie.value
    for i in range(prop):
        mmTrans[i, i] = mmTrans[i,i] + lamda_val * num_row
    movieTrans=broad_movie.value.T    
    rateTrans=u_rate_Trans[x, :].T
    upd_movie = movieTrans * rateTrans
    
    return np.linalg.solve(mmTrans, upd_movie)

#For Fixing the user matrix
def fix_user_mat(x):
    uuTrans = broad_user.value.T * broad_user.value
    for a in range(prop):
        uuTrans[a, a] = uuTrans[a,a] + lamda_val * num_col

    userTrans=broad_user.value.T   
    ratingTrans=broad_rating_mat.value[x,:].T    
    upd_user = userTrans * ratingTrans

    return np.linalg.solve(uuTrans, upd_user)
   
if __name__ == "__main__":
    
    num_itr =  10   
    rms_val = np.zeros(num_itr)
    lamda_val = 0.001
    prop = 15
    sc = SparkContext(appName="recmsys_als")
   
# Getting the movielens data
    inputData = sc.textFile(sys.argv[1])
    inputLines = inputData.map(lambda line: line.split(","))
    
    #rating_mat = np.matrix(inputLines.collect()).astype('float')
    line_array = np.array(inputLines.collect()).astype('float')
    rating_mat = np.matrix(line_array)

    num_row,num_col = rating_mat.shape
    
    broad_rating_mat = sc.broadcast(rating_mat)
      
# To randomly initialize movie matrix and user matrix
    movie_mat = matrix(rand(num_row, prop))
    broad_movie = sc.broadcast(movie_mat)
    
    user_mat = matrix(rand(num_col, prop))
    broad_user = sc.broadcast(user_mat)
   
    movie_mat_row,movie_mat_col = movie_mat.shape
    user_mat_row,user_mat_col = user_mat.shape

# iterating until movie matrix and user matrix converges
    for i in range(0,num_itr):
    
        #Fixing the user matrix for finding the movie matrix. 
        movie_mat = sc.parallelize(range(movie_mat_row)).map(fix_user_mat).collect()
        broad_movie = sc.broadcast(matrix(np.array(movie_mat)[:, :]))

        #Fixing the movie matrix for finding the user matrix.
        user_mat = sc.parallelize(range(user_mat_row)).map(fix_movie_mat).collect()
        broad_user = sc.broadcast(matrix(np.array(user_mat)[:, :]))

        rms_error_val = get_rms_error(rating_mat, matrix(np.array(movie_mat)), matrix(np.array(user_mat)))
        rms_val[i] = rms_error_val
    fin_user_mat = np.array(user_mat).squeeze()
    fin_movie_mat = np.array(movie_mat).squeeze()
    
    fin_out = np.dot(fin_movie_mat,fin_user_mat.T)

# For Initializing the weights matrix 
    weight_mat = np.zeros(shape=(num_row,num_col))
    for r in range(num_row):
        for j in range(num_col):
            if rating_mat[r,j]>= 0.5:
                weight_mat[r,j] = 1.0
            else:
                weight_mat[r,j] = 0.0
    
# subtract the rating that user has rated
    rate_max=5
    movie_recom = np.argmax(fin_out - rate_max * weight_mat,axis =1)
    
# To Predict the movie for each user
    for u in range(0, movie_mat_row):
        r = movie_recom.item(u)
        p = fin_out.item(u,r)
        print ('The movie predicted for user_id %d: for movie_id %d: Predicted rating is %f ' %(u+1,r+1,p) )
        
    print "RMSE value after each iterations: ",rms_val    
    print "Avg rmse---- ",np.mean(rms_val)
    sc.stop()
