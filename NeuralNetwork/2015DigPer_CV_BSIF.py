'''
	By Stanley Gan, email:glgan@sfu.ca
	Cross validation on model based on BSIF features  and fine tuning model
	1) val_accuracy with PCA 0.995
	2) val_accuracy without PCA 0.99
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
train_real_bsif = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Train_Real_DigPerson.txt',sep='\s+',header=None)
train_fake_bsif = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Train_Spoof_DigPerson.txt',sep='\s+',header=None)

numRealTrain = int(pd.to_numeric(train_real_bsif.shape[0]))
numFakeTrain = int(pd.to_numeric(train_fake_bsif.shape[0]))

####################### For bsif ############################
#Concat real and fake training data and label targets
training_bsif = pd.concat([train_real_bsif,train_fake_bsif])
setTarg_train_bsif = np.concatenate( (np.ones((numRealTrain,1)) , np.zeros((numFakeTrain,1)) ),axis=0)
training_bsif['target'] = setTarg_train_bsif

#handling some data here
numFeatures_bsif = int(pd.to_numeric(training_bsif.shape[1]))
feature_train_bsif = training_bsif.iloc[:,0:numFeatures_bsif]
target_train_bsif = training_bsif['target']

feature_train_bsif = feature_train_bsif.values
feature_train_bsif = preprocessing.scale(feature_train_bsif)	#normalize
target_train_bsif = target_train_bsif.values


############## Start model and ML here ########################

#Using cross validation to see performance 
folds=10
skf = StratifiedKFold(n_splits=folds, shuffle=True)
num_ep = 50
batchSize=50
gaus_sigma=3
do=0.3

print("\nInformation about validation on the BSIF model:")
sum_acc_single2 = 0
i=1
val_loss_single2 = np.zeros((1,num_ep))
train_loss_single2 = np.zeros((1,num_ep))

for train, validate in skf.split(feature_train_bsif, target_train_bsif):
	print "Running fold ", i, " over ", folds, " folds"
	
	#dimensionality reduction, PCA
	pca = PCA()

	trainFeat = pca.fit_transform(feature_train_bsif[train])
	validateFeat = pca.transform(feature_train_bsif[validate])
	trainFeat = feature_train_bsif[train]
	validateFeat = feature_train_bsif[validate]
 
	numFeatures_bsif = trainFeat.shape[1]
     
	#Define model
	model = None
	model = Sequential()
	model.add(Dense(numFeatures_bsif,input_dim=numFeatures_bsif,init='glorot_uniform',activation='relu'))
	model.add(GaussianNoise(gaus_sigma))
	model.add(Dropout(do))
	#model.add(Dense(numFeatures_bsif, activation='relu'))
	#model.add(Dropout(do))
	#model.add(Dense(numFeatures_bsif, activation='relu'))
	#model.add(Dropout(do))
	model.add(Dense(1,activation='sigmoid'))

	#compile model
	opt = Adadelta()
	model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

	#Fit and evaluate model
	hist = model.fit(trainFeat, target_train_bsif[train], nb_epoch=num_ep, batch_size=batchSize, verbose=0, validation_data=[validateFeat, target_train_bsif[validate]], shuffle=True)

	#Some details for plot later
	sum_acc_single2 = sum_acc_single2 + hist.history['val_acc'][num_ep-1]
	train_loss_single2 = train_loss_single2 + hist.history['loss']
	val_loss_single2 = val_loss_single2 + hist.history['val_loss']
	i=i+1

mean_acc_single2 = sum_acc_single2/folds
val_loss_single2 = val_loss_single2/folds
train_loss_single2 = train_loss_single2/folds
print("Mean of accuracy for bsif model on " + str(folds) + " folds validation data: " + str(mean_acc_single2))

#Print loss graph for bsif model
plt.figure(3)
plt.plot(np.arange(1,num_ep+1),train_loss_single2[0])
plt.plot(np.arange(1,num_ep+1),val_loss_single2[0])
plt.title("Average training loss and average validation loss for " + str(folds) + "\n folds across epochs for BSIF feature extraction model")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','validation'])
plt.show()



