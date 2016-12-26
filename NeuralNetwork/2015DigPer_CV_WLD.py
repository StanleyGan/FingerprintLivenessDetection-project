'''

	By Stanley Gan, email:glgan@sfu.ca
	Cross validation on model based on WLD features and fine tune it
	1) Fine tune val_acc: 0.9955
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
train_real_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Train_Real_DigPerson.txt',sep='\s+',header=None)
train_fake_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Train_Spoof_DigPerson.txt',sep='\s+',header=None)

numRealTrain = int(pd.to_numeric(train_real_wld.shape[0]))
numFakeTrain = int(pd.to_numeric(train_fake_wld.shape[0]))

####################### For lpq ############################
#Concat real and fake training data and label targets
training_wld = pd.concat([train_real_wld,train_fake_wld])
setTarg_train_wld = np.concatenate( (np.ones((numRealTrain,1)) , np.zeros((numFakeTrain,1)) ),axis=0)
training_wld['target'] = setTarg_train_wld

#handling some data here
numFeatures_wld = int(pd.to_numeric(training_wld.shape[1]))
print(numFeatures_wld)
feature_train_wld = training_wld.iloc[:,0:numFeatures_wld]
target_train_wld = training_wld['target']

feature_train_wld = feature_train_wld.values
feature_train_wld = preprocessing.scale(feature_train_wld)	#normalize
target_train_wld = target_train_wld.values


############## Start model and ML here ########################

#Using cross validation to see performance 
folds=10
skf = StratifiedKFold(n_splits=folds, shuffle=True)
num_ep = 50
batchSize=50
gaus_sigma=1
do=0.5

print("\nInformation about validation on the WLD model:")
sum_acc_single2 = 0
i=1
val_loss_single2 = np.zeros((1,num_ep))
train_loss_single2 = np.zeros((1,num_ep))

for train, validate in skf.split(feature_train_wld, target_train_wld):
	print "Running fold ", i, " over ", folds, " folds"

	#dimensionality reduction, PCA
	pca = PCA()

	trainFeat = pca.fit_transform(feature_train_wld[train])
	validateFeat = pca.transform(feature_train_wld[validate])
	
	trainFeat = feature_train_wld[train]
	validateFeat = feature_train_wld[validate]
	numFeatures_wld = trainFeat.shape[1]

	#define model
	model = None
	model = Sequential()
	model.add(Dense(numFeatures_wld,input_dim=numFeatures_wld,init='glorot_uniform',activation='relu'))
	model.add(GaussianNoise(gaus_sigma))
	model.add(Dropout(do))
	model.add(Dense(numFeatures_wld, activation='relu'))
	model.add(Dropout(do))
	#model.add(Dense(numFeatures_wld, activation='relu'))
	#model.add(Dropout(do))
	model.add(Dense(1,activation='sigmoid'))

	#compile model
	opt = Adadelta()
	model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

	#Fit and evaluate model
	hist = model.fit(trainFeat, target_train_wld[train], nb_epoch=num_ep, batch_size=batchSize, verbose=0, validation_data=[validateFeat, target_train_wld[validate]], shuffle=True)

	#Some details for plot later
	sum_acc_single2 = sum_acc_single2 + hist.history['val_acc'][num_ep-1]
	train_loss_single2 = train_loss_single2 + hist.history['loss']
	val_loss_single2 = val_loss_single2 + hist.history['val_loss']
	i=i+1

mean_acc_single2 = sum_acc_single2/folds
val_loss_single2 = val_loss_single2/folds
train_loss_single2 = train_loss_single2/folds
print("Mean of accuracy for WLD model on " + str(folds) + " folds validation data: " + str(mean_acc_single2))

#Print loss graph for WLD model
plt.figure(3)
plt.plot(np.arange(1,num_ep+1),train_loss_single2[0])
plt.plot(np.arange(1,num_ep+1),val_loss_single2[0])
plt.title("Average training loss and average validation loss for " + str(folds) + "\n folds across epochs for WLD feature extraction model")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','validation'])
plt.show()



