import timeit
import pandas as pd
import numpy as np

trainfile = './data/training.csv'
testfile = './data/sorted_test.csv'
submissionfile = './data/sample_submission.csv'
xtrain = pd.DataFrame()
xtest = pd.DataFrame()
submission = pd.DataFrame()
targetCols = ['Ca','P','pH','SOC','Sand']
feature_list = {}


def loaddata():
	global xtrain, xtest, submission
	xtrain = pd.read_csv(trainfile,tupleize_cols=True) #these are the training features
	xtest = pd.read_csv(testfile) #these are thre training answers
	submission = pd.read_csv(submissionfile)
	ytrain = xtrain[targetCols]
	xtrain.drop(targetCols,axis=1,inplace=True)
	print ("Finished Loading Data ....xtrain:%s, ytrain:%s xtest:%s, submission:%s " % (str(xtrain.shape), str(ytrain.shape), str(xtest.shape), str(submission.shape)))
	
def cleandata():
	global xtrain, xtest, submission
	print "finished cleaning the data"

def transformdata():
	global xtrain, xtest, submission
	print "finised transforming the data"

def featureselection():
	global xtrain, xtest, submission
	for target in targetCols:
		selector = SelectKBest(f_regression, k=2500)
		selector.fit_transform(x_train[spectra_features], y_train[target])
		selected = selector.get_support()
		feats = [col for (col,sel) in zip(list(x_train[spectra_features].columns.values), selected) if sel]
		feature_list[target] = feats 

def crossvalidation():
	cv = cross_validation.KFold(len(x_train), n_folds=10, indices=False, shuffle=True)

def main():
    print "starting..."
    loaddata()   #Loads the raw data from the csv files into pandas.dataframes
    cleandata()	 #modifies xtrain, ytrain by removing noise
    transformdata() #modifies xtrain, ytrain by Transforming data to smooth
    featureselection() #modifies xtrain, ytrain by removing features that are not being considered

    print "finished..."
#LOAD

#DATA CLEANING

#DATA TRANSFORMATION

#CV

#MODEL

#EVALUATION

#OUTPUT
if __name__ == '__main__':
	main()