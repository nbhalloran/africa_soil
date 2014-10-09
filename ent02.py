import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", RuntimeWarning) 

import numpy as np
import utilities as util
import sklearn.linear_model as linear
import sklearn.ensemble as ensemble
from sklearn import cross_validation
import pandas as pd
import sys
import evaluation as ev
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import BayesianRidge, LinearRegression
from scipy import signal
import matplotlib.pyplot as plt
import re
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
from scipy.ndimage import filters as flt
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import math
import correlation_based_smoothing as CBS
import cookb_signalsmooth as cbss 
import random


issub = False
#cvtype = "standard"
cvtype = "nick"
#Linux
trainloc = "/home/nick/Documents/kaggle/africa_soil/data/training.csv"
testloc = "/home/nick/Documents/kaggle/africa_soil/data/sorted_test.csv"
predloc = "/home/nick/Documents/kaggle/africa_soil/data/predictions_new.csv"
sampleloc = "/home/nick/Documents/kaggle/africa_soil/data/sample_submission.csv"
#Windows
trainloc = r"C:\DS\kaggle\projects\africa_soil\data\training.csv"
testloc = r"C:\DS\kaggle\projects\africa_soil\data\sorted_test.csv"
predloc = r"C:\DS\kaggle\projects\africa_soil\data\predictions.csv"
sampleloc = r"C:\DS\kaggle\projects\africa_soil\data\sample_submission.csv"



def beatthebenchmark():
	predloc = "/home/nick/Documents/kaggle/africa_soil/data/beatthebenchmark1.csv"
	train = pd.read_csv(trainloc)
	test = pd.read_csv(testloc)
	labels = train[['Ca','P','pH','SOC','Sand']].values

	train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
	test.drop('PIDN', axis=1, inplace=True)

	xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]
	print xtrain.shape

	sup_vec = svm.SVR(C=11000.0, verbose = 2)

	preds = np.zeros((xtest.shape[0], 5))
	for i in range(5):
	    sup_vec.fit(xtrain, labels[:,i])
	    preds[:,i] = sup_vec.predict(xtest).astype(float)

	sample = pd.read_csv(sampleloc)
	sample['Ca'] = preds[:,0]
	sample['P'] = preds[:,1]
	sample['pH'] = preds[:,2]
	sample['SOC'] = preds[:,3]
	sample['Sand'] = preds[:,4]

	sample.to_csv(predloc, index = False)


def btbCrossValidation():
	predloc = "/home/nick/Documents/kaggle/africa_soil/data/beatthebenchmark.csv"
	train = pd.read_csv(trainloc)
	test = pd.read_csv(trainloc)
	labels = train[['Ca','P','pH','SOC','Sand']].values

	train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
	test.drop('PIDN', axis=1, inplace=True)

	xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]

	sup_vec = svm.SVR(C=10000.0)#, verbose = 2)
	

	cv = cross_validation.KFold(len(train), n_folds=2, indices=False, shuffle=False)
	subresults = {}
	results = []

	for i in range(5):
		if i== 0:
			print "got here SOC"
			feats[i] = list(xtrain[2300:2499].columns) + list(xtrain[2500:2720].columns) + list(xtrain[3150:3500].columns) #+ non_spectra_feats_inc
		elif i==1:
			print "got here Sand"
			feats[i] = list(xtrain.columns[2220:2420]) + list(xtrain.columns[3175:3375]) + list(xtrain.columns[3420:3540]) + non_spectra_feats
		elif i==2:
			print "got here pH"
			feats[i] = list(xtrain.columns[1150:1200]) + list(xtrain.columns[1940:2000]) + list(xtrain.columns[2000:2070]) + \
					list(xtrain.columns[2120:2210]) + list(xtrain.columns[2950:3070]) + list(xtrain.columns[3100:3250]) + \
					list(xtrain.columns[3400:3500])  
			 #+ non_spectra_feats
		elif i==3:
			print "got here P"
			feats[i] = list(xtrain.columns[0:300]) + list(xtrain.columns[3100:3270]) + list(xtrain.columns[2800:3000])
			feats[i] = list(xtrain.columns[25:275]) + list(xtrain.columns[2880:2930])
			#feats = non_spectra_feats
		elif i==4:
			print "got here Ca"
			feats[i] = list(xtrain.columns[1150:1210]) + list(xtrain.columns[2560:2610]) + list(xtrain.columns[2870:2930]) + \
					list(xtrain.columns[2970:3050]) + list(xtrain.columns[3120:3230]) + list(xtrain.columns[3231:3340]) + \
					list(xtrain.columns[3550:3570]) + non_spectra_feats
		 


	for train_sub, test_sub in cv:
		preds = np.zeros((xtest[test_sub].shape[0], 5))
		for i in range(5):	
			print("gothere", i)
			sup_vec.fit(xtrain[feats[i]][train_sub], labels[:,i][train_sub])
			preds[:,i] = sup_vec.predict(xtest[test_sub]).astype(float)
			subresults[i] = ev.rmse(np.array(labels[:,i])[test_sub],np.array(preds[:,i]))
		subtotal = 0 
		for x in subresults:
			subtotal = subtotal + subresults[x]
		print ("average for the run is ", subtotal/5)
		results.append(subtotal/5)
	print "Results: " + str( np.array(results).mean() )


def nickmain1():


	targets = ['Ca','P','pH','SOC','Sand']
	C02_band = ['m2379.76','m2377.83','m2375.9','m2373.97','m2372.04','m2370.11','m2368.18','m2366.26','m2364.33','m2362.4','m2360.47','m2358.54','m2356.61','m2354.68','m2352.76']
	train_cols_to_remove = ['PIDN']+targets+C02_band
	#use the below to tune one at a time
	#targets = ['P']

	df_train = pd.read_csv(trainloc,tupleize_cols=True)
	df_test =  pd.read_csv(testloc)
	x_train=df_train.drop(train_cols_to_remove,axis=1)
	y_train=df_train[targets]

	train_feature_list = list(x_train.columns)
	spectra_features = train_feature_list

	non_spectra_feats=['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI','Depth'] 
	non_spectra_feats_inc=['BSAN','BSAS','BSAV','REF7','TMAP','Depth'] 
	#non_spectra_feats_inc=['TMAP','Depth'] 
	for feats in non_spectra_feats:
 		spectra_features.remove(feats)

 	fltSpectra=flt.gaussian_filter1d(np.array(x_train[spectra_features]),sigma=20,order=1)
	#sets the subsoil variable depth to be either 1 or 0
	x_train["Depth"] = x_train["Depth"].apply(lambda depth:0 if depth =="Subsoil" else 1)

	tvals = {}
	counter = 0 
	x_train["xmap"] = 0

	for x in x_train['TMAP'].unique():
		x_train['xmap'][x_train['TMAP']==x] = counter
		counter = counter + 1

	# 	if x in tvals:
	# 		x_train["xmap"] = counter
	# 	else:
	# 		tvals[x] = x
	# 		counter = counter + 1
	# 		x_train["xmap"] = counter
			
	# for x

	



	# x = x_train[spectra_features].ix[0]
	# plt.plot(x)
	# y = cbss.smooth(x, window_len=100, window='hanning')
	# plt.plot(y)
	# plt.show()
	# counter = 0 

	# x_train = x_train[spectra_features+non_spectra_feats]

	# for row_index, row  in x_train.iterrows():
	# 	x = x_train.ix[row_index]
	# 	plt.plot(x)
	

	# for row_index, row  in x_train.iterrows():
	# 	x = x_train.ix[row_index]
	# 	x_train.ix[row_index]= cbss.smooth(x_train.ix[row_index], window_len=2, window='flat')
	# 	plt.plot(x_train.ix[row_index])

	# plt.show()
	

#	for x in range(0,1000):
#		x_train[spectra_features].ix[x].plot()

	#plt.plot(x_train[spectra_features].ix[0])
	#x_train[spectra_features]=fltSpectra
	#plt.plot(x_train[spectra_features].ix[0])
	#plt.show()

	# plt.plot(y_train["Ca"])
	# plt.plot(y_train["P"])
	# plt.show()
	# return

	
	# y = math.log(x)
	# print y
	# print math.exp(y)

	
	
	#y_train["P"] = y_train["P"].apply(lambda x: math.log(x+10))
	

	#SELECT Kth BEST FEATURES TO INCLUDE
	feats_list = {}
	for target in targets:
		selector = SelectKBest(f_regression, k=5)
		selector.fit_transform(x_train[spectra_features], y_train[target])
		selected = selector.get_support()
		
		if target == "SOCx":
			print "got here SOC"
			feats = list(x_train[spectra_features].columns[2300:2499]) + list(x_train[spectra_features].columns[2500:2720]) + list(x_train[spectra_features].columns[3150:3500]) #+ non_spectra_feats_inc
		elif target == "Sandx":
			print "got here Sand"
			feats = list(x_train[spectra_features].columns[2220:2420]) + list(x_train[spectra_features].columns[3175:3375]) + list(x_train[spectra_features].columns[3420:3540]) + non_spectra_feats
		elif target == "pHx":
			print "got here pH"
			feats = list(x_train[spectra_features].columns[1150:1200]) + list(x_train[spectra_features].columns[1940:2000]) + list(x_train[spectra_features].columns[2000:2070]) + \
					list(x_train[spectra_features].columns[2120:2210]) + list(x_train[spectra_features].columns[2950:3070]) + list(x_train[spectra_features].columns[3100:3250]) + \
					list(x_train[spectra_features].columns[3400:3500])  
			 #+ non_spectra_feats
		elif target == "Px":
			print "got here P"
			feats = list(x_train[spectra_features].columns[0:300]) + list(x_train[spectra_features].columns[3100:3270]) + list(x_train[spectra_features].columns[2800:3000])
			feats = list(x_train[spectra_features].columns[25:275]) + list(x_train[spectra_features].columns[2880:2930])
			#feats = non_spectra_feats
		elif target == "Cax":
			print "got here Ca"
			feats = list(x_train[spectra_features].columns[1150:1210]) + list(x_train[spectra_features].columns[2560:2610]) + list(x_train[spectra_features].columns[2870:2930]) + \
					list(x_train[spectra_features].columns[2970:3050]) + list(x_train[spectra_features].columns[3120:3230]) + list(x_train[spectra_features].columns[3231:3340]) + \
					list(x_train[spectra_features].columns[3550:3570]) + non_spectra_feats
		else:
			print "hmmm"
			feats = [col for (col,sel) in zip(list(x_train[spectra_features].columns.values), selected) if sel]
			feats = list(x_train[spectra_features].columns[0:3578])
		
		#feats_list[target] = list(x_train[spectra_features].columns[0:3578])  + non_spectra_feats_inc
		
		feats_list[target] = feats + non_spectra_feats_inc
		feats_list[target] = feats 

		# print feats[0:1000]
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m5243.56'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m5214.63'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m2534.03'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m2497.39'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m1893.78'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m1866.78'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m1681.64'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m1627.64'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m1407.8'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m1280.52'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m1203.38'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m1064.53'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m607.474'))
		# print ("here it is" , x_train[spectra_features].columns.get_loc('m603.617'))






	if issub == False:
		if cvtype == "standard": 
			cv = cross_validation.KFold(len(x_train), n_folds=10, indices=False, shuffle=True)
			subresults = {}
			results = []
			for train_sub, test_sub in cv:
				for target in targets:
					#clf = ensemble.GradientBoostingRegressor(n_estimators=6)
					#clf = RandomForestRegressor(n_estimators = 40)
					#clf = linear_model.Lasso(alpha=0.08)
					#clf = svm.SVC(C=10000.0, verbose = 2)
					clf = svm.SVR(C=1000.0)#, verbose = 2)
					#clf = tree.DecisionTreeRegressor(min_samples_leaf=10000)
					#clf = Ridge(alpha=1.0)
					#clf = ElasticNet(alpha=0.1, l1_ratio=0.7)
					#clf = BayesianRidge(compute_score=False, normalize=True, n_iter=300,fit_intercept=False)
					clf.fit(np.array(x_train[feats_list[target]])[train_sub], np.array(y_train[target])[train_sub])
					pred = clf.predict(np.array(x_train[feats_list[target]])[test_sub]).astype(float)
					subresults[target] = ev.rmse(np.array(y_train[target])[test_sub],np.array(pred))
					#df[target] = pred
					

				subtotal = 0 
				for x in subresults:
					subtotal = subtotal + subresults[x]
				print ("average for the run is ", subtotal/len(targets))
				results.append(subtotal/len(targets))
			print "Results: " + str( np.array(results).mean() )
		else:
			print "non standard CV"

			

			subresults = {}
			results = []
			for zzz in range(15):
				x_train["cvtrain"] = True
				for x in random.sample(xrange(0,len(x_train['TMAP'].unique()) - 1), 10):
					x_train["cvtrain"][x_train["xmap"]==x] = False
				for target in targets:
					
					clf = svm.SVR(C=10000.0)#, verbose = 2)
					clf.fit(np.array(x_train[feats_list[target]])[x_train["cvtrain"] == True], np.array(y_train[target])[x_train["cvtrain"] == True])
					pred = clf.predict(np.array(x_train[feats_list[target]])[x_train["cvtrain"] == False]).astype(float)
					subresults[target] = ev.rmse(np.array(y_train[target])[x_train["cvtrain"] == False],np.array(pred))
					#df[target] = pred
					

				subtotal = 0 
				for x in subresults:
					subtotal = subtotal + subresults[x]
				print ("average for the run is ", subtotal/len(targets))
				results.append(subtotal/len(targets))
			print "Results: " + str( np.array(results).mean() )

	else:
		test_all = pd.read_csv(testloc)
		#fltSpectra=flt.gaussian_filter1d(np.array(test_all[spectra_features]),sigma=20,order=1)
		#test_all[spectra_features]=fltSpectra	
		test_all["Depth"] = test_all["Depth"].apply(lambda depth:0 if depth =="Subsoil" else 1)
		ids = np.genfromtxt(testloc, dtype=str, skip_header=1, delimiter=',', usecols=0)
		df = pd.DataFrame({"PIDN": ids, "Ca": test_all['PIDN'], "P": test_all['PIDN'], "pH": test_all['PIDN'], "SOC": test_all['PIDN'], "Sand": test_all['PIDN']})
		for target in targets:
			#clf = ensemble.GradientBoostingRegressor(n_estimators=6)
			#clf = RandomForestRegressor(n_estimators = 20)
			#clf = linear_model.Lasso(alpha=0.08)
			#clf = svm.SVC()
			clf = svm.SVR(C=5000.0)
			#clf = tree.DecisionTreeRegressor(min_samples_leaf=20)
			#clf = Ridge(alpha=1.0)
			#clf = ElasticNet(alpha=0.1, l1_ratio=0.7)
			#clf = BayesianRidge(compute_score=False, normalize=True, n_iter=300,fit_intercept=False)
			clf.fit(np.array(x_train[feats_list[target]]), np.array(y_train[target]))
			pred = clf.predict(np.array(test_all[feats_list[target]]))
			df[target] = pred
			df.to_csv(predloc, index=False, cols=["PIDN","Ca","P","pH","SOC","Sand"])




def smoothListGaussian(list,degree=5):  
	window=degree*2-1  
	weight=np.array([1.0]*window)  
	weightGauss=[]  
	for i in range(window):  
		i=i-degree+1  
		frac=i/float(window)  
		gauss=1/(np.exp((4*(frac))**2))  
		weightGauss.append(gauss)  
	weight=np.array(weightGauss)*weight  
	#smoothed=[0.0]*(len(list)-window)  
	smoothed=[0.0]*(len(list))  
	for i in range(len(smoothed)):  
		#print i, i+window, len(list)
		if i+window > len(list):
			smoothed[i]= list[i]# sum(np.array(list[i:len(list)-1])*weight)/sum(weight)
		else:
			smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)

	return smoothed  

def main():
    #nickmain1()
    #btbCrossValidation()
    beatthebenchmark()
    
if __name__ == '__main__':
    main()




