'''
	By Stanley Gan, email:glgan@sfu.ca
	Cross valdidation on model using LPQ features and finetune it
	1) Finetune and got 1.0 val_acc
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, GaussianNoise
from keras.optimizers import SGD, Adadelta
from keras.engine.topology import Merge

#Load features
train_real_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Train_Real_DigPerson.txt',sep='\s+',header=None)
train_fake_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Train_Spoof_DigPerson.txt',sep='\s+',header=None)

numRealTrain = int(pd.to_numeric(train_real_lpq.shape[0]))
numFakeTrain = int(pd.to_numeric(train_fake_lpq.shape[0]))

####################### For lpq ############################
#Concat real and fake training data and label targets
training_lpq = pd.concat([train_real_lpq,train_fake_lpq])
setTarg_train_lpq = np.concatenate( (np.ones((numRealTrain,1)) , np.zeros((numFakeTrain,1)) ),axis=0)
training_lpq['target'] = setTarg_train_lpq

#handling some data here
numFeatures_lpq = int(pd.to_numeric(training_lpq.shape[1]))
print(numFeatures_lpq)
feature_train_lpq = training_lpq.iloc[:,0:numFeatures_lpq]
target_train_lpq = training_lpq['target']

feature_train_lpq = feature_train_lpq.values
feature_train_lpq = preprocessing.scale(feature_train_lpq)	#normalize
target_train_lpq = target_train_lpq.values


############## Start model and ML here ########################

#Using cross validation to see performance 
folds=10
skf = StratifiedKFold(n_splits=folds, shuffle=True)
num_ep = 50
batchSize=50
gaus_sigma=1
do=0.3

print("\nInformation about validation on the LPQ model:")
sum_acc_single2 = 0
i=1
val_loss_single2 = np.zeros((1,num_ep))
train_loss_single2 = np.zeros((1,num_ep))

for train, validate in skf.split(feature_train_lpq, target_train_lpq):
	print "Running fold ", i, " over ", folds, " folds"
	
	#dimensionality reduction, PCA
	pca = PCA()

	trainFeat = pca.fit_transform(feature_train_lpq[train])
	validateFeat = pca.transform(feature_train_lpq[validate])
	trainFeat = feature_train_lpq[train]
	validateFeat = feature_train_lpq[validate]

	numFeatures_lpq = trainFeat.shape[1]
	print(numFeatures_lpq)

	#define model
	model = None
	model = Sequential()
	model.add(Dense(numFeatures_lpq,input_dim=numFeatures_lpq,init='glorot_uniform',activation='relu'))
	model.add(GaussianNoise(gaus_sigma))
	#model.add(Dropout(do))
	#model.add(Dense(numFeatures_lpq, activation='relu'))
	#model.add(Dropout(do))
	#model.add(Dense(numFeatures_lpq, activation='relu'))
	model.add(Dropout(do))
	model.add(Dense(1,activation='sigmoid'))

	#compile model
	opt = Adadelta()
	model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

	#Fit and evaluate model
	hist = model.fit(trainFeat, target_train_lpq[train], nb_epoch=num_ep, batch_size=batchSize, verbose=0, validation_data=[validateFeat, target_train_lpq[validate]], shuffle=True)

	#Some details for plot later
	sum_acc_single2 = sum_acc_single2 + hist.history['val_acc'][num_ep-1]
	train_loss_single2 = train_loss_single2 + hist.history['loss']
	val_loss_single2 = val_loss_single2 + hist.history['val_loss']
	i=i+1

mean_acc_single2 = sum_acc_single2/folds
val_loss_single2 = val_loss_single2/folds
train_loss_single2 = train_loss_single2/folds
print("Mean of accuracy for LPQ model on " + str(folds) + " folds validation data: " + str(mean_acc_single2))

#Print loss graph for LPQ model
plt.figure(3)
plt.plot(np.arange(1,num_ep+1),train_loss_single2[0])
plt.plot(np.arange(1,num_ep+1),val_loss_single2[0])
plt.title("Average training loss and average validation loss for " + str(folds) + "\n folds across epochs for LPQ feature extraction model")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','validation'])
plt.show()



