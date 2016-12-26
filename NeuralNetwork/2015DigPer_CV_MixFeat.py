'''
	By Stanley Gan, email:glgan@sfu.ca
	Cross validate for model selection after trying out certain models, here I utilize PCA on dataset as it gave a great improvement
	1) without pca 0.993
	2) with pca 0.995
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

#Merges 3 models where each of them uses different features i.e. bsif, lpq, wld, you could add extra ones here if you want to
def concatFeatModel(numFeatures_bsif, numFeatures_lpq, numFeatures_wld):
	# Model for bsif
	model_bsif = Sequential()
	model_bsif.add(Dense(numFeatures_bsif,input_dim=numFeatures_bsif,init='glorot_uniform',activation='relu'))
	model_bsif.add(GaussianNoise(3))
	model_bsif.add(Dropout(0.3))

	# Model for lpq
	model_lpq = Sequential()
	model_lpq.add(Dense(numFeatures_lpq,input_dim=numFeatures_lpq,init='glorot_uniform',activation='relu'))
	model_lpq.add(GaussianNoise(3))
	model_lpq.add(Dropout(0.3))

	# Model for wld
	model_wld = Sequential()
	model_wld.add(Dense(numFeatures_wld,input_dim=numFeatures_wld,init='glorot_uniform',activation='relu'))
	model_wld.add(GaussianNoise(3))
	model_wld.add(Dropout(0.3))
	model_wld.add(Dense(numFeatures_wld, activation='relu'))
	model_wld.add(Dropout(0.3))

	#Merge all models
	model_concat = Sequential()
	model_concat.add(Merge([model_bsif, model_lpq, model_wld],mode='concat',concat_axis=1))
	model_concat.add(Dense(1,activation='sigmoid'))

	#compile model
	opt = Adadelta()
	model_concat.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

	return model_concat

#Load features
train_real_bsif = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Train_Real_DigPerson.txt',sep='\s+',header=None)
train_fake_bsif = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Train_Spoof_DigPerson.txt',sep='\s+',header=None)
test_real_bsif = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Test_Real_DigPerson.txt',sep='\s+',header=None)
test_fake_bsif = pd.read_csv('./data/Data_2015_BSIF_7_12_motion_Test_Spoof_DigPerson_1500f.txt',sep='\s+',header=None)

train_real_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Train_Real_DigPerson.txt',sep='\s+',header=None)
train_fake_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Train_Spoof_DigPerson.txt',sep='\s+',header=None)
test_real_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Test_Real_DigPerson.txt',sep='\s+',header=None)
test_fake_lpq = pd.read_csv('./data/Data_2015_LPQ_3_11_motion_Test_Spoof_DigPerson_1500f.txt',sep='\s+',header=None)

train_real_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Train_Real_DigPerson.txt',sep='\s+',header=None)
train_fake_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Train_Spoof_DigPerson.txt',sep='\s+',header=None)
test_real_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Test_Real_DigPerson.txt',sep='\s+',header=None)
test_fake_wld = pd.read_csv('./data/Data_2015_WLD_3_8_motion_Test_Spoof_DigPerson_1500f.txt',sep='\s+',header=None)

#all features contains the same number of samples as we are extracting from identitcal samples. num of features different
numRealTrain = int(pd.to_numeric(train_real_bsif.shape[0]))
numFakeTrain = int(pd.to_numeric(train_fake_bsif.shape[0]))
numRealTest = int(pd.to_numeric(test_real_bsif.shape[0]))
numFakeTest = int(pd.to_numeric(test_fake_bsif.shape[0]))

###################### For bsif ################################

#Concat real and fake training data, PCA fit it and label targets
training_bsif = pd.concat([train_real_bsif,train_fake_bsif])
setTarg_train_bsif = np.concatenate( (np.ones((numRealTrain,1)) , np.zeros((numFakeTrain,1)) ),axis=0)
training_bsif['target'] = setTarg_train_bsif

#concat real and fake test data, transform it with PCA for training data, label targets
test_bsif = pd.concat([test_real_bsif, test_fake_bsif])
setTarg_test_bsif = np.concatenate( (np.ones((numRealTest,1)) , np.zeros((numFakeTest,1)) ),axis=0)
test_bsif['target'] = setTarg_test_bsif

#handling some data here
numFeatures_bsif = int(pd.to_numeric(training_bsif.shape[1]))
feature_train_bsif = training_bsif.iloc[:,0:numFeatures_bsif]
target_train_bsif = training_bsif['target']
feature_test_bsif = test_bsif.iloc[:,0:numFeatures_bsif]
target_test_bsif = test_bsif['target']

feature_train_bsif = feature_train_bsif.values
feature_train_bsif = preprocessing.scale(feature_train_bsif)	#normalize
target_train_bsif = target_train_bsif.values
feature_test_bsif = feature_test_bsif.values
feature_test_bsif = preprocessing.scale(feature_test_bsif)
target_test_bsif = target_test_bsif.values

####################### For lpq ############################

#Concat real and fake training data, PCA fit it and label targets
training_lpq = pd.concat([train_real_lpq,train_fake_lpq])
setTarg_train_lpq = np.concatenate( (np.ones((numRealTrain,1)) , np.zeros((numFakeTrain,1)) ),axis=0)
training_lpq['target'] = setTarg_train_lpq

#concat real and fake test data, transform it with PCA for training data, label targets
test_lpq = pd.concat([test_real_lpq, test_fake_lpq])
setTarg_test_lpq = np.concatenate( (np.ones((numRealTest,1)) , np.zeros((numFakeTest,1)) ),axis=0)
test_lpq['target'] = setTarg_test_lpq

#handling some data here
numFeatures_lpq = int(pd.to_numeric(training_lpq.shape[1]))
feature_train_lpq = training_lpq.iloc[:,0:numFeatures_lpq]
target_train_lpq = training_lpq['target']
feature_test_lpq = test_lpq.iloc[:,0:numFeatures_bsif]
target_test_lpq = test_lpq['target']

feature_train_lpq = feature_train_lpq.values
feature_train_lpq = preprocessing.scale(feature_train_lpq)	#normalize
target_train_lpq = target_train_lpq.values
feature_test_lpq = feature_test_lpq.values
feature_test_lpq = preprocessing.scale(feature_test_lpq)
target_test_lpq = target_test_lpq.values

########################## For wld #############################

#Concat real and fake training data, PCA fit it and label targets
training_wld = pd.concat([train_real_wld,train_fake_wld])
setTarg_train_wld = np.concatenate( (np.ones((numRealTrain,1)) , np.zeros((numFakeTrain,1)) ),axis=0)
training_wld['target'] = setTarg_train_wld

#concat real and fake test data, transform it with PCA for training data, label targets
test_wld = pd.concat([test_real_wld, test_fake_wld])
setTarg_test_wld = np.concatenate( (np.ones((numRealTest,1)) , np.zeros((numFakeTest,1)) ),axis=0)
test_wld['target'] = setTarg_test_wld

#handling some data here
numFeatures_wld = int(pd.to_numeric(training_wld.shape[1]))
feature_train_wld = training_wld.iloc[:,0:numFeatures_wld]
target_train_wld = training_wld['target']
feature_test_wld = test_wld.iloc[:,0:numFeatures_wld]
target_test_wld = test_wld['target']

feature_train_wld = feature_train_wld.values
feature_train_wld = preprocessing.scale(feature_train_wld)	#normalize
target_train_wld = target_train_wld.values
feature_test_wld = feature_test_wld.values
feature_test_wld = preprocessing.scale(feature_test_wld)
target_test_wld = target_test_wld.values

############## Start model and ML here ########################

#Using cross validation to see performance of different models on validation data
folds=10
skf = StratifiedKFold(n_splits=folds, shuffle=True)
num_ep = 50
batchSize=50

print("\nInformation about validation on the concat model:\n")
sum_acc_concat = 0
i=1
val_loss_concat = np.zeros((1,num_ep))
train_loss_concat = np.zeros((1,num_ep))

#Validate on concat model
for train, validate in skf.split(feature_train_bsif, target_train_bsif):
	print "Running fold ", i, " over ", folds, " folds"

	#Dimensionality reduction using PCA
	pca = PCA()
	transf_feature_train_bsif = pca.fit_transform(feature_train_bsif[train])
	#print(transf_feature_train_bsif.shape)
	transf_feature_val_bsif = pca.transform(feature_train_bsif[validate])
	numFeatures_bsif = transf_feature_train_bsif.shape[1]

	transf_feature_train_lpq = pca.fit_transform(feature_train_lpq[train])
	#print(transf_feature_train_lpq.shape)
	transf_feature_val_lpq = pca.transform(feature_train_lpq[validate])
	numFeatures_lpq = transf_feature_train_lpq.shape[1]

	transf_feature_train_wld = pca.fit_transform(feature_train_wld[train])
	#print(transf_feature_train_wld.shape)
	transf_feature_val_wld = pca.transform(feature_train_wld[validate])
	numFeatures_wld = transf_feature_train_wld.shape[1]

	#create model
	model_concat = None	#refresh
	model_concat = concatFeatModel(numFeatures_bsif, numFeatures_lpq, numFeatures_wld)

	hist = model_concat.fit([transf_feature_train_bsif, transf_feature_train_lpq, transf_feature_train_wld], target_train_bsif[train], nb_epoch=num_ep, batch_size=batchSize, verbose=0, validation_data=[[transf_feature_val_bsif, transf_feature_val_lpq, transf_feature_val_wld], target_train_bsif[validate]], shuffle=True)

	sum_acc_concat = sum_acc_concat + hist.history['val_acc'][num_ep-1]
	train_loss_concat = train_loss_concat + hist.history['loss']
	val_loss_concat = val_loss_concat + hist.history['val_loss']
	i=i+1

mean_acc_concat = sum_acc_concat/folds
val_loss_concat = val_loss_concat/folds
train_loss_concat = train_loss_concat/folds
print("Mean of accuracy for concat model on " + str(folds) + " folds validation data: " + str(mean_acc_concat) + "\n")

#Plot loss graph for concat model
plt.figure(1)
plt.plot(np.arange(1,num_ep+1),train_loss_concat[0])
plt.plot(np.arange(1, num_ep+1),val_loss_concat[0])
plt.title("Average training loss and average validation loss for " + str(folds) + "\n folds across epochs on Concat Model")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','validation'])
plt.show()



