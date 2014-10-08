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
from scipy.ndimage import filters 


issub = False
trainloc = "/home/nick/Documents/kaggle/africa_soil/data/training.csv"
testloc = "/home/nick/Documents/kaggle/africa_soil/data/sorted_test.csv"
predloc = "/home/nick/Documents/kaggle/africa_soil/data/predictions.csv"

def beatthebenchmark():
	#Columns to be picked from training file
	pickTrain = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI','Depth','Ca','P','pH','SOC','Sand']
	data = np.genfromtxt(trainloc, names=True, delimiter=',', usecols=(pickTrain))
	#Column to be picked from test file
	pickTest = ['PIDN', 'BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
	test = np.genfromtxt(testloc, names=True, delimiter=',', usecols=(pickTest))
	ids = np.genfromtxt(testloc, dtype=str, skip_header=1, delimiter=',', usecols=0)
	#Features to train model on
	featuresList = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
	#Keep a copy of train file for later use
	data1 = np.copy(data)
	#Dependent/Target variables
	targets = ['Ca','P','pH','SOC','Sand']
	#Prepare empty result
	df = pd.DataFrame({"PIDN": ids, "Ca": test['PIDN'], "P": test['PIDN'], "pH": test['PIDN'], "SOC": test['PIDN'], "Sand": test['PIDN']})
	for target in targets:
		#Prepare data for training
		data, testa, features, fillVal = util.prepDataTrain(data1, target, featuresList, False, 10, False, True, 'mean', False, 'set')
	print 'Data preped'
	#Use/tune your predictor
	clf = ensemble.GradientBoostingRegressor(n_estimators=20)
	clf.fit(data[features].tolist(), data[target])
	#Prepare test data
	test = util.prepDataTest(test, featuresList, True, fillVal, False, 'set')
	#Get predictions
	pred = clf.predict(test[features].tolist())
	#Store results
	df[target] = pred
	df.to_csv(predloc, index=False, cols=["PIDN","Ca","P","pH","SOC","Sand"])

def nickmain1():

	train_all = pd.read_csv(trainloc)
	target_all = pd.read_csv(trainloc)
	test_all = pd.read_csv(testloc)
	targets = ['Ca','P','pH','SOC','Sand']
	train_cols_to_remove = ['PIDN']+targets
	train_all["Depth"] = train_all["Depth"].replace(["Topsoil", "Subsoil"],[10,-10])
	test_all["Depth"] = test_all["Depth"].replace(["Topsoil", "Subsoil"],[10,-10])
	common_features = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
	feats_list = {}
	colnames_nums = []
	colnames = train_all.ix[:,'m7497.96':'m599.76'].columns.values
	for x in colnames:
		match = re.search(r'(?<=m)[0-9]*',x)
		if match: 
			colnames_nums.append(int(match.group()))
	
	print len(colnames)
	print len(colnames_nums)
	print len(train_all.ix[0,'m7497.96':'m599.76'].values)


	

	for target in targets:
		selector = SelectKBest(f_regression, k=200)
		selector.fit_transform(train_all.ix[:,'m7497.96':'m599.76'], train_all[target])
		selected = selector.get_support()
		feats = [col for (col,sel) in zip(list(train_all.ix[:,'m7497.96':'m599.76'].columns.values), selected) if sel]
		feats_list[target] = feats+common_features

		


	#pickTest = ['PIDN', 'BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI','Depth']#ORIGINAL10
	ids = np.genfromtxt(testloc, dtype=str, skip_header=1, delimiter=',', usecols=0)
	df = pd.DataFrame({"PIDN": ids, "Ca": test_all['PIDN'], "P": test_all['PIDN'], "pH": test_all['PIDN'], "SOC": test_all['PIDN'], "Sand": test_all['PIDN']})
	
	cv = cross_validation.KFold(len(train_all), n_folds=10, indices=False)
	subresults = {}
	results = []

	if issub == False:
		for train_sub, test_sub in cv:
			for target in targets:
				#clf = ensemble.GradientBoostingRegressor(n_estimators=6)
				#clf = RandomForestRegressor(n_estimators = 40)
				#clf = linear_model.Lasso(alpha=0.08)
				#clf = svm.SVC()
				#clf = tree.DecisionTreeRegressor(min_samples_leaf=20)
				#clf = Ridge(alpha=1.0)
				#clf = ElasticNet(alpha=0.1, l1_ratio=0.7)
				clf = BayesianRidge(compute_score=True)
				clf.fit(np.array(train_all[feats_list[target]])[train_sub], np.array(train_all[target])[train_sub])
				pred = clf.predict(np.array(train_all[feats_list[target]])[test_sub])
				subresults[target] = ev.rmse(np.array(train_all[target])[test_sub],np.array(pred))
				#df[target] = pred
			subtotal = 0 
			for x in subresults:
				subtotal = subtotal + subresults[x]
			print ("average for the run is ", subtotal/len(targets))
			results.append(subtotal/len(targets))
		print "Results: " + str( np.array(results).mean() )

	else:
		for target in targets:
			#clf = ensemble.GradientBoostingRegressor(n_estimators=6)
			#clf = RandomForestRegressor(n_estimators = 20)
			#clf = linear_model.Lasso(alpha=0.08)
			#clf = svm.SVC()
			#clf = tree.DecisionTreeRegressor(min_samples_leaf=20)
			#clf = Ridge(alpha=1.0)
			#clf = ElasticNet(alpha=0.1, l1_ratio=0.7)
			clf = BayesianRidge(compute_score=True)
			clf.fit(np.array(train_all[feats_list[target]]), np.array(train_all[target]))
			pred = clf.predict(np.array(test_all[feats_list[target]]))
			df[target] = pred
			df.to_csv(predloc, index=False, cols=["PIDN","Ca","P","pH","SOC","Sand"])

#Main

def testGauss(x, y, s, npts):
    b = gaussian(39, 10)
    ga = filters.convolve1d(y, b/b.sum())
    plt.plot(x, ga)
    print "gaerr", ssqe(ga, s, npts)
    return ga

def testButterworth(nyf, x, y, s, npts):
    b, a = butter(4, 1.5/nyf)
    fl = filtfilt(b, a, y)
    plt.plot(x,fl)
    print "flerr", ssqe(fl, s, npts)
    return fl    

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
    nickmain1()

    
if __name__ == '__main__':
    main()




#DONEimport evaluation
#DONE calculate RMSE on each column and then get the average find the evaluation type
#DONE Getcrossvalidationcode
#DONE modularizecrossvalidationcode
#DONE getcrossvalidationscore  Results: 0.986888854843
#DONE getkagglescore 0.92905
#DONE usethe regression diagram
#DONE pick 3 types of regression and implement
#DONE get the best one
#DONEsplit into functions
#DONEfigure out how to set all of the topsoil to 1 and subsoil to 2
#DONEget 10 best univariate feature selection
#DONEget score
#DONEget 15 best univatiate feature selection
#DONEget score
