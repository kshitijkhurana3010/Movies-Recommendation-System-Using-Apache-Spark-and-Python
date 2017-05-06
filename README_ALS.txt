'''
Cloud Computing Project ALS implementation
Group 21
Team: Satish Kumar, Abhijit Nair, Kshitij Khurana
'''

Execution Steps
============================


Step 1: Run the below commnd to get the initial rating matrix.
        

        $ python init_matrix_eval.py ratings.csv  init_matrix.csv

Step 2: To do Initial Setup
	
	$ sudo su cloudera
	$ hadoop fs -mkdir /user/cloudera/recsys /user/cloudera/recsys/input

Step 3. Put all the input files into the new input directory
	
        $ hadoop fs -put init_matrix.csv /user/cloudera/recsys/input

Step 4. Execute the source code

        $ spark-submit recmsys_als.py /user/cloudera/recsys/input/init_matrix.csv > outputals.txt

Step 4. Delete the input file to place different input file

        $ hadoop fs -rm -r /user/cloudera/recsys
	

OUTPUT:-

The movie predicted for user_id 1: for movie_id 186: Predicted rating is 0.163824 
The movie predicted for user_id 2: for movie_id 147: Predicted rating is 3.303564 
The movie predicted for user_id 3: for movie_id 403: Predicted rating is 1.460827 
The movie predicted for user_id 4: for movie_id 436: Predicted rating is 2.606832 
The movie predicted for user_id 5: for movie_id 100: Predicted rating is 2.045087 
The movie predicted for user_id 6: for movie_id 481: Predicted rating is 0.659329 
The movie predicted for user_id 7: for movie_id 50: Predicted rating is 2.137729 


RMSE value after each iterations:  [ 0.42151727  0.37606982  0.36963965  0.36780211  0.36695056  0.36646097
  0.36614617  0.36592943  0.36577298  0.36565622]

Avg rmse----  0.373194518223


