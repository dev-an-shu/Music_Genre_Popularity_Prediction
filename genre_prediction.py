# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:40:10 2019

@author: Devanshu
"""
#11:34 
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

import time

#Converting HDFS file to CSV (Pre Run)
def HDFS_to_CSV(path,output_file):
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
			
			if first_run:
				song_df.to_csv(output_file)
				first_run = False
			else:
				with open(output_file, 'a', encoding='utf-8') as fl:
					song_df.to_csv(fl, header=False)
			store.close()
	return df

#Function to apply feature selecction (using random forest) on training dataset 
def applyFeatureSelection(X_train, Y_train):
	#Feature Selection using Random Forest (Gini Index Method)
	from sklearn.ensemble import RandomForestRegressor
	
	#Refrence dataset for calculating feature importance
	X = X_train #Training dataset
	Y = Y_train #Label (Class values)
	
	#Feature selection model
	feature_selection_model = RandomForestRegressor(n_estimators=10,max_depth=40)
	feature_selection_model.fit(X,Y)
	
	#Features and their gini index
	features = X.columns
	feature_importances = feature_selection_model.feature_importances_
	
	features_df = pd.DataFrame({'Features': features, 'Importance Score': feature_importances})
	features_df.sort_values('Importance Score', inplace=True, ascending=False)
	#print("Feature Importance: ",features_df)
#	features_df.plot(x='Features',y='Importance Score',kind='bar')
#	pl.suptitle("Important Features")
	
	#Return top 30 models
	X_top_features = X[list(features_df.head(30).Features)]
	
	return X_top_features, Y

#Function to get class level accuracy for genre model (With 10 genres)
def classLevelAccuracy(target, predict):
	
	# Target Label class count
	target = np.array(target)
	unique, count = np.unique(target, return_counts=True)
	targetCount = dict(zip(unique, count))

	#Initalizing predicted class count for each class
	class0CorrectCount = 0
	class1CorrectCount = 0
	class2CorrectCount = 0
	class3CorrectCount = 0
	class4CorrectCount = 0
	class5CorrectCount = 0
	class6CorrectCount = 0
	class7CorrectCount = 0
	class8CorrectCount = 0
	class9CorrectCount = 0
	
	#The final accuracy list which will contain accuracy for each class
	accuracyList = []
	
	# Predicted class value count
	for j in range(len(predict)):
		if target[j] == 0 and predict[j] == target[j]:
			class0CorrectCount += 1
		elif target[j] == 1 and predict[j] == target[j]:
			class1CorrectCount += 1
		elif target[j] == 2 and predict[j] == target[j]:
			class2CorrectCount += 1
		elif target[j] == 3 and predict[j] == target[j]:
			class3CorrectCount += 1
		elif target[j] == 4 and predict[j] == target[j]:
			class4CorrectCount += 1
		elif target[j] == 5 and predict[j] == target[j]:
			class5CorrectCount += 1
		elif target[j] == 6 and predict[j] == target[j]:
			class6CorrectCount += 1
		elif target[j] == 7 and predict[j] == target[j]:
			class7CorrectCount += 1
		elif target[j] == 8 and predict[j] == target[j]:
			class8CorrectCount += 1
		elif target[j] == 9 and predict[j] == target[j]:
			class9CorrectCount += 1
	
	#Calculate  accuracy of each class
	for className, classCount in targetCount.items():
		if className == 0:
			accuracyList.append(class0CorrectCount / classCount)
		elif className == 1:
			accuracyList.append(class1CorrectCount / classCount)
		elif className == 2:
			accuracyList.append(class2CorrectCount / classCount)
		elif className == 3:
			accuracyList.append(class3CorrectCount / classCount)
		elif className == 4:
			accuracyList.append(class4CorrectCount / classCount)
		elif className == 5:
			accuracyList.append(class5CorrectCount / classCount)
		elif className == 6:
			accuracyList.append(class6CorrectCount / classCount)
		elif className == 7:
			accuracyList.append(class7CorrectCount / classCount)
		elif className == 8:
			accuracyList.append(class8CorrectCount / classCount)
		elif className == 9:
			accuracyList.append(class9CorrectCount / classCount)
		
	return accuracyList

#Function to Up sample (using SMOTE) the training dataset 
def samplingData(X_train, Y_train):
	#SMOTE (Synthetic Minority Over-sampling Technique)
	#SMOTE is an over-sampling method. It creates synthetic samples of the minority class. We use imblearn python package to over-sample the minority classes
	sm = SMOTE()
	X_train, Y_train = sm.fit_sample(X_train, Y_train)
	
	return X_train, Y_train

#Perform Stratified Cross validation on passed dataset with feature selectiona and sampling
def getClassLevelAccuracyWithSCV(model, X, Y, cv):
	#Stratified Cross validation
	skf = StratifiedKFold(n_splits=cv)
	
	classValueAcc = []
	
	#Get Straified folds
	for train_index, test_index in skf.split(X, Y):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		
		#Apply Feature selection on training dataset of each fold
		X_train, Y_train = applyFeatureSelection(X_train, Y_train)
		X_train = pd.DataFrame(X_train)
		
		#To check class distribution before sampling
		labelClass = np.array(Y_train)
		unique_elements, counts_elements = np.unique(labelClass, return_counts=True)
		print("Before Sampling\n",np.asarray((unique_elements, counts_elements)))
		topColumns = X_train.columns
		
		#Sampling Training dataset
		X_train, Y_train  = samplingData(X_train,Y_train)
		
		#To check class distribution after sampling
		labelClass = np.array(Y_train)
		unique_elements, counts_elements = np.unique(labelClass, return_counts=True)
		print("After Sampling\n",np.asarray((unique_elements, counts_elements)))
		
		#Training Dataset
		model.fit(X_train, Y_train)
		
		#Select top features
		print("Selected Columns: ",topColumns)
		X_test = X_test[list(topColumns)]
		
		pred = model.predict(X_test)
		#Calcualte class level accuracy using classLevelAccuracy function
		classValueAcc.append(classLevelAccuracy(Y_test, pred))
		
	return classValueAcc

def main():
	start_time = time.time()
	# Generate CSV file from raw hdfs
	#HDFS_to_CSV(path,output_file)
	
	# Get data from generated CSV
	filename = "./msd_genre_dataset.csv"
	song_genre_data_df = pd.read_csv(filename,index_col=0)
	
	print("Number of Rows: ",len(song_genre_data_df))
	print("Number of Columns: ",len(song_genre_data_df.columns))
	#print("Feature's Data type:\n",song_genre_data_df.dtypes)
	
	#Heat Map to get a visualization of feature correlation
	corr = song_genre_data_df.corr()
	plt.figure(figsize = (7,7))
	sns.heatmap(corr)
	
	#print("Class Values: ",song_genre_data_df["genre"].unique())
	#print("Class Value Count:\n",song_genre_data_df["genre"].value_counts())
	
	#Convert label to category type which will help us endcode class values
	song_genre_data_df["genre"] = song_genre_data_df["genre"].astype("category")
	
	#Creating a new feature called songGenre by giving number values to different class values
	#Eg: classic pop and rock - 0, folk - 1, dance and electronica - 2 and so on
	song_genre_data_df["songGenre"] = song_genre_data_df["genre"].cat.codes
	
	#Percentage representage of class value in the entire dataset
	ClassCount = song_genre_data_df['genre'].value_counts()
	print("ClassCount\n",ClassCount / len(song_genre_data_df) * 100)
	
	#Split dataset into training and testing dataset with 70/30 ratio
	from sklearn.model_selection import train_test_split
	X = song_genre_data_df.drop(['genre','songGenre'],axis=1) #removing labels
	X = X.select_dtypes(exclude=[object]) #converting labels into object datatype
	Y = song_genre_data_df['songGenre'].values
	
	#General Split to test the model with dataset
	trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.70)
	
	#To check class value distrbution
	labelClass = np.array(trainY)
	unique_elements, counts_elements = np.unique(labelClass, return_counts=True)
	
	#Before Sampling class distribution (Note: Class Value have been encoded)
	print("Before Sampling\n",np.asarray((unique_elements, counts_elements)))
	
	#SMOTE (Synthetic Minority Over-sampling Technique)
	#SMOTE is an over-sampling method. It creates synthetic samples of the minority class. We use imblearn python package to over-sample the minority classes
	sm = SMOTE()
	
	#############################Random Forest as classifier with 100 tress per forest
	from sklearn.ensemble import RandomForestClassifier
	model_rf = RandomForestClassifier(n_estimators=100)
	
	#Up Sampliing training dataset using SMOTE
	trainX, trainY = sm.fit_sample(trainX, trainY)
	
	#To check class distribution after up sampling 
	labelClass = np.array(trainY)
	unique_elements, counts_elements = np.unique(labelClass, return_counts=True)
	print("After Sampling\n",np.asarray((unique_elements, counts_elements)))
	
	#Train training dataset on random forest model
	model_rf.fit(trainX, trainY)
	#Prediction using random forest
	pred_rf = model_rf.predict(testX)
	
	#Get class value accuracy after applying random forest model
	classAcc = classLevelAccuracy(testY, pred_rf)
	print("Random Forest Class Level Accuracy\n",classAcc)
	
	#Get Overall accuracy using random forest model
	accuracy_rf = accuracy_score(testY, pred_rf)
	print("RF Accuracy: ", accuracy_rf)
	
	#Stratified Cross validation with feature selection and sampling for Random Forest
	cv_scores_rf2 = getClassLevelAccuracyWithSCV(model_rf, X, Y, cv=2)
	
	#Take mean of 2 folds
	cv_scores_rf2 = np.array(cv_scores_rf2)
	fold1 = cv_scores_rf2[0]
	fold2 = cv_scores_rf2[1]
	classAccRF2 = []
	classAccRF2Display = []
	classAccRF2Display.append("RF Class Level Accuracy")
	for i in range(len(fold1)):
		classMean = (fold1[i] + fold2[i]) / 2
		classAccRF2.append(classMean)
		classAccRF2Display.append(classMean)
		
	print("RF Accuracy after 2 Fold Stratified Cross Validation:\n", classAccRF2)
	
	
	
	#############################KNN classifier with 100 neighbours
	from sklearn import neighbors
	model_knn = neighbors.KNeighborsClassifier(n_neighbors=100, weights='uniform')
	
	#General Split to test the model with dataset
	trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.70)
	
	#Apply Up Sampling using SMOTE
	trainX, trainY = sm.fit_sample(trainX, trainY)
	
	#Training dataset
	model_knn.fit(trainX, trainY)
	
	#Predicting
	pred_knn = model_knn.predict(testX)
	
	#Get class value accuracy after applying KNN model
	classAcc = classLevelAccuracy(testY, pred_knn)
	print("KNN Class Level Accuracy\n",classAcc)
	
	#Get Overall accuracy using KNN model
	accuracy_knn = accuracy_score(testY, pred_knn)
	print("KNN Accuracy: ", accuracy_knn)
	
	#Stratified Cross validation with feature selection and sampling for KNN
	cv_scores_knn2 = getClassLevelAccuracyWithSCV(model_knn, X, Y, cv=2)
	
	#Take mean of 2 folds
	cv_scores_knn2 = np.array(cv_scores_knn2)
	fold1 = cv_scores_knn2[0]
	fold2 = cv_scores_knn2[1]
	classAccKNN2 = []
	
	#For display purpose
	classAccKNN2Display = []
	classAccKNN2Display.append("KNN Class Level Accuracy")
	for i in range(len(fold1)):
		classMean = (fold1[i] + fold2[i]) / 2
		classAccKNN2.append(classMean)
		classAccKNN2Display.append(classMean)
		
	print("KNN Accuracy after 2 Fold Cross Validation\n", classAccKNN2)
	
	
	
	#############################SVM as classifier with linear kenral
	from sklearn.svm import LinearSVC
	from sklearn.preprocessing import StandardScaler
	model_svm = LinearSVC()
	
	#Scalling Dataset for improving SVM model
	scaler = StandardScaler()
	scaler.fit_transform(X)
	
	#General Split to test the model with dataset
	trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.70)
	
	#Apply Up Sampling using SMOTE
	trainX, trainY = sm.fit_sample(trainX, trainY)
	
	#Training Dataset
	model_svm.fit(trainX, trainY)
	
	#Predicting 
	pred_svm = model_svm.predict(testX)
	
	#Get class value accuracy after applying SVM model
	classAcc = classLevelAccuracy(testY, pred_svm)
	print("SVM Class Level Accuracy\n",classAcc)
	
	#Get Overall accuracy using SVM model
	accuracy_svm = accuracy_score(testY, pred_svm)
	print("SVM Accuracy: ", accuracy_svm)
	
	#Stratified Cross validation with feature selectiona and sampling for SVM
	cv_scores_svm2 = getClassLevelAccuracyWithSCV(model_svm, X, Y, cv=2)
	
	#Taking mean of 2 folds
	cv_scores_svm2 = np.array(cv_scores_svm2)
	fold1 = cv_scores_svm2[0]
	fold2 = cv_scores_svm2[1]
	classAccSVM2 = []
	
	#For display purpose
	classAccSVM2Display = []
	classAccSVM2Display.append("SVM Class Level Accuracy")
	for i in range(len(fold1)):
		classMean = (fold1[i] + fold2[i]) / 2
		classAccSVM2.append(classMean)
		classAccSVM2Display.append(classMean)
		
	print("SVM Accuracy after 2 Fold Cross Validation\n", classAccSVM2)
	
	
	
	
	#General Result Comparision
	generalResult = {'Model':['SVM','KNN','RF'],'Accuracy':[accuracy_svm, accuracy_knn, accuracy_rf]}
	agg_scores = pd.DataFrame(data = generalResult)
	agg_scores.plot(x='Model',y='Accuracy',kind='bar')
	pl.suptitle("Result Comparision After Applying Up Sampling")
	
	end_time = time.time()
	
	print("Execution Time: ",(end_time - start_time))
	#Output Data Frame
	allResults = pd.DataFrame({
	    'class':['Class','classic pop and rock', 'punk', 'folk', 'pop', 'dance and electronica', 'metal', 'jazz and blues', 'classical', 'hip-hop', 'soul and reggae'],
	    'SVM Class Level Accuracy':classAccSVM2Display,
	    'KNN Class Level Accuracy':classAccKNN2Display,
	    'RF  Class Level Accuracy':classAccRF2Display
	})
	#Create Result folder with timestamp(yyyy-mm-dd-HH-MM-SS) and generate result.csv
	import datetime
	timestr = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
	folderName = "Result For Genre Classification "+ timestr #Name of folder
	os.mkdir(folderName)
	folderPath = Path(folderName)
	outFileName = "Genre_Result_With_CV_FS_Sampling.csv" #Name of result file
	outArray = allResults
	outArray.to_csv(Path(folderPath, outFileName), index = None, header=False)
	
	
if __name__ == '__main__':
	main()