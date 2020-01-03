# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:12:59 2019
@author: Devanshu
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from pathlib import Path
import pylab as pl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import time 

#Converting HDFS file to CSV (Pre Run)
def HDFS_to_CSV(path,output_file,return_df=False):
	base_dir = path
	ext = ".H5"
	df = pd.DataFrame()
	first_run = True
	for root,dirs,files in os.walk(base_dir):
		files = glob.glob(os.path.join(root,"*"+ext))
		for f in files:
			print("File: ",f)
			store = pd.HDFStore(f)
			song_analysis = pd.read_hdf(store,'/analysis/songs')
			metadata = pd.read_hdf(store,'/metadata/songs')
			musicbrainz = pd.read_hdf(store,'musicbrainz/songs')
			frames = [song_analysis,metadata,musicbrainz]
			song_df = pd.concat(frames,axis=1)
			print("song_df created")
			
			if return_df:
				df.append(song_df)
				
			if first_run:
				song_df.to_csv(output_file)
				first_run = False
			else:
				with open(output_file, 'a', encoding='utf-8') as fl:
					song_df.to_csv(fl, header=False)
			store.close()
	if return_df:
		return df

#Function to read CSV file
def get_songs_df_from_csv(filename):
    return pd.read_csv(filename,index_col=0)   

#Function to remove missing data
def removeMissingness(df):
	df = df.fillna(0)
	
    #Drop rows with year = 0
	df = df.drop(df[df.year == 0].index)
    
	#Drop rows with song hotness = nan
	df = df.drop(df[df.song_hotttnesss==""].index)
	
	#Odering Indices
	indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
	
	return df[indices_to_keep]

#Function to apply feature selecction (using random forest) on training dataset
def applyFeatureSelection(X_train, Y_train):
	#Feature Importance and Reduction
	from sklearn.ensemble import RandomForestRegressor
	
	#Feature selection model	
	feature_selection_model = RandomForestRegressor(n_estimators=10,max_depth=40)
	feature_selection_model.fit(X_train,Y_train)
	
	#Features and their gini index
	features = X_train.columns
	feature_importances = feature_selection_model.feature_importances_
	
	features_df = pd.DataFrame({'Features': features, 'Importance Score': feature_importances})
#	print(features_df)
	features_df.sort_values('Importance Score', inplace=True, ascending=False)
#	features_df.plot(x='Features',y='Importance Score',kind='bar')
	
	#Split dataset into training and testing
	X_top_features = X_train[list(features_df.head().Features)]
	
	return X_top_features, Y_train

#Perform Stratified Cross validation on passed dataset with feature selectiona
def applyStratifiedCrossvalidation(model,X,Y,cv):
	#Stratified Cross validation
	skf = StratifiedKFold(n_splits=cv)
	
	SCVAcc = []
	
	#Get Straified folds
	for train_index, test_index in skf.split(X, Y):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		
		#Apply Feature selection on training dataset of each fold
		X_train, Y_train = applyFeatureSelection(X_train, Y_train)
		
		#Training Dataset
		model.fit(X_train, Y_train)
		
		#Select top features
		X_train = pd.DataFrame(X_train)
		topColumns = X_train.columns
		print("Selected Columns: ",topColumns)
		X_test = X_test[list(topColumns)]
		
		#Prediction and calcualte accuracy
		pred = model.predict(X_test)
		foldAcc = accuracy_score(Y_test, pred)
		SCVAcc.append(foldAcc)
		
	return SCVAcc
	
def	main():
	start_time = time.time()
	#Getting data from files
#	path = '../MillionSongSubset/data'
#	output_file = 'msd_simple.csv'
	
	#HDFS_to_CSV(path,output_file,return_df=True) 	
	
	#Getting data from generated CSV
	songs_df = get_songs_df_from_csv('./msd_popularity_dataset.csv')
	songs_df.reset_index(drop=True,inplace=True)
	print("Number of Rows: ",len(songs_df))
	print("Number of Columns: ",len(songs_df.columns))
	
	#Removing missing data
	print("Starting Data Clean")
	df = removeMissingness(songs_df)
	
	#Feature Correlation
	corr = df.corr()
	plt.figure(figsize = (7,7))
	sns.heatmap(corr)
	
	#Year comparision with tempo, duration and loudness
	agg_df = df.groupby(df.year).mean().reset_index()
	agg_df.plot(x='year',y='tempo')
	agg_df.plot(x='year',y='duration')
	agg_df.plot(x='year',y='loudness')
	
	#Creating a new feature called isPopular based on if the song hotness is above mean or not
	song_hotness_mean = df.song_hotttnesss.mean()
	threshold = song_hotness_mean
	df['isPopular'] = 0
	df = df.set_value(df[df.song_hotttnesss>threshold].index,'isPopular',1)
	
	#Getting dataset and label class
	X = df.drop(['song_hotttnesss','isPopular'],axis=1)
	X = X.select_dtypes(exclude=[object])
	Y = df['isPopular'].values
	
	
	
	#############################SVM with linear kernel
	from sklearn.svm import SVC
	model_svm = SVC(kernel='linear')
	
	#General Split to test the model with dataset
	trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.70)
	
	#Training Dataset
	model_svm.fit(trainX, trainY)
	
	#Prediction
	pred = model_svm.predict(testX)
	
	#Calculating accuracy
	accuracy_svm = accuracy_score(testY, pred)
	print("SVM Accuracy: ",accuracy_svm)
	
	#Stratified Cross validation with feature selection for SVM
	scores_svm2 = applyStratifiedCrossvalidation(model_svm,X,Y,cv=2)
	scores_svm4 = applyStratifiedCrossvalidation(model_svm,X,Y,cv=4)
	scores_svm5 = applyStratifiedCrossvalidation(model_svm,X,Y,cv=5)
	print("SVM after 2 fold stratified cross validation: ",mean(scores_svm2))
	print("SVM after 4 fold stratified cross validation: ",mean(scores_svm4))
	print("SVM after 5 fold stratified cross validation: ",mean(scores_svm5))
	
	
	#############################KNN with 100 neighbours
	from sklearn import neighbors
	model_knn = neighbors.KNeighborsClassifier(n_neighbors=100, weights='uniform')
	
	#General Split to test the model with dataset
	trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.70)
	
	#Training Dataset
	model_knn.fit(trainX,trainY)
	
	#Prediction
	pred = model_knn.predict(testX)
	
	#Calcualting accuracy
	accuracy_knn = accuracy_score(testY,pred)
	print("KNN Accuracy: ",accuracy_knn)
	
	#Stratified Cross validation with feature selection for KNN
	scores_knn2 = applyStratifiedCrossvalidation(model_knn,X,Y,cv=2)
	scores_knn4 = applyStratifiedCrossvalidation(model_knn,X,Y,cv=4)
	scores_knn5 = applyStratifiedCrossvalidation(model_knn,X,Y,cv=5)
	print("KNN after 2 fold stratified cross validation: ",mean(scores_knn2))
	print("KNN after 4 fold stratified cross validation: ",mean(scores_knn4))
	print("KNN after 5 fold stratified cross validation: ",mean(scores_knn5))
	
	
	#############################Random Forest with 100 trees
	from sklearn.ensemble import RandomForestClassifier
	model_rf = RandomForestClassifier(n_estimators=100)
	
	#General Split to test the model with dataset
	trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.70)
	
	#Training Dataset
	model_rf.fit(trainX, trainY)
	
	#Prediction
	pred = model_rf.predict(testX)
	
	#Calculating Accuracy
	accuracy_rf = accuracy_score(testY, pred)
	print("Random Forest Accuracy: ",accuracy_rf)
	
	#Stratified Cross validation with feature selection for Random Forest
	scores_rf2 = applyStratifiedCrossvalidation(model_rf,X,Y,cv=2)
	scores_rf4 = applyStratifiedCrossvalidation(model_rf,X,Y,cv=4)
	scores_rf5 = applyStratifiedCrossvalidation(model_rf,X,Y,cv=5)
	print("Random forest after 2 fold stratified cross Validation: ",mean(scores_rf2))
	print("Random forest after 4 fold stratified cross Validation: ",mean(scores_rf4))
	print("Random forest after 5 fold stratified cross Validation: ",mean(scores_rf5))
	
	
	
	#General Result Comparision
	d = {'Model':['SVM','KNN','RF'],'Accuracy':[accuracy_svm, accuracy_knn, accuracy_rf]}
	agg_scores = pd.DataFrame(data = d)
	agg_scores.plot(x='Model',y='Accuracy',kind='bar')
	pl.suptitle("Model Result Comparision")
	
	#Stratified Cross validation comparision
	d = {'Model':['SVM-2','SVM-4','SVM-5','KNN-2','KNN-4','KNN-5','RF-2','RF-4','RF-5'],'Accuracy':[mean(scores_svm2), mean(scores_svm4), mean(scores_svm5), mean(scores_knn2), mean(scores_knn4), mean(scores_knn5) ,mean(scores_rf2), mean(scores_rf4), mean(scores_rf5)]}
	agg_scores = pd.DataFrame(data = d)
	agg_scores.plot(x='Model',y='Accuracy',kind='bar')
	pl.suptitle("Stratified Cross Validation Result Comparision")
	
	end_time = time.time()
	
	print("Execurtion Time: ",(end_time - start_time))
	#Output Data Frame
	allResults = pd.DataFrame({'Model':['Model-CV','SVM-2','SVM-4','SVM-5','KNN-2','KNN-4','KNN-5','RF-2','RF-4','RF-5'],'Accuracy':['Accuracy', mean(scores_svm2), mean(scores_svm4), mean(scores_svm5), mean(scores_knn2), mean(scores_knn4), mean(scores_knn5) ,mean(scores_rf2), mean(scores_rf4), mean(scores_rf5)]})
	
	#Create Result folder with timestamp(yyyy-mm-dd-HH-MM-SS) and generate result.csv
	import datetime
	timestr = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
	folderName = "Result For Popularity Classification "+ timestr #Name of folder
	os.mkdir(folderName)
	folderPath = Path(folderName)
	outFileName = "Popularity_Result_With_CV_FS.csv" #Name of result file
	outArray = allResults
	outArray.to_csv(Path(folderPath, outFileName), index = None, header=False)

if __name__ == '__main__':
	main()