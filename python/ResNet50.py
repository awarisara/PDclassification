import numpy as np # linear algebra
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

pd = np.load('PDsample_files',allow_pickle=True)
ctrl = np.load('Controlsample_files',allow_pickle=True)

print('PD shape',pd.shape)
print('Control shape',ctrl.shape)

pd_label = np.repeat(np.array([[0,1]]),number of PD sample, axis = 0) 
ctrl_label = np.repeat(np.array([[1,0]]),number of Control sample, axis = 0) 

#resolution of spectrogram
m = 300
n = 39

sample = np.concatenate([pd,ctrl],axis=0)
sample = sample.reshape(amount of all samples, m,n,1): # PD + Control samples
label = np.concatenate([pd_label,ctrl_label],axis=0)
print(sample.shape)
print(label.shape)

#X_train, X_test, y_train, y_test = train_test_split(sample, label, test_size=0.2, random_state=42)
#print('Xtrain:',X_train.shape)
#print('Xtest:',X_test.shape)
#print('ytrain:',y_train.shape)
#print('ytest:',y_test.shape)

inputs = sample
targets = label

#https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
acc_per_fold = []
loss_per_fold = []
pric_per_fold = []
sens_per_fold = []
spec_per_fold = []
f1_per_fold = []
verbosity = 1
num_folds = 10
no_epochs = 100
batch_size = 32

# Merge inputs and targets
#inputs = np.concatenate((X_train, X_test), axis=0)
#targets = np.concatenate((y_train, y_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

	np.save('X_train'+str(fold_no)+'.npy',inputs[train])
	np.save('X_test'+str(fold_no)+'.npy',inputs[test])
	np.save('y_train'+str(fold_no)+'.npy',targets[train])
	np.save('y_test'+str(fold_no)+'.npy',targets[test])
	
	model = ResNet50(weights=None, include_top=False,input_shape=(m,n,1))
	x = model.output
	#Global Average Pooling is a pooling operation designed to replace fully connected layers in classical CNNs
	#data_format: default = channels_last that corresponds to inputs with shape (batch, height, width, channels)
	#If keepdims is False (default), the rank of the tensor is reduced for spatial dimensions. output shape is (batch_size, channels)
	x = GlobalAveragePooling2D(name='avg_pool')(x)
	x = Dropout(0.4)(x)
	predictions = Dense(2, activation='softmax')(x)
	model = Model(inputs=model.input, outputs=predictions)
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
	model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

	#Generate a print
	print('------------------------------------------------------------------------')
	print(f'Training for fold {fold_no} ...')

	#save model
	checkpoint_path = "training_cp_"+str(fold_no)+".ckpt"
	#checkpoint_dir = os.path.dirname("/data/users/fengdcw/30039nocut/resnet/")

	# Create a callback that saves the model's weights
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
	
	#Fit data to model
	callback = EarlyStopping(monitor='val_loss', mode="min",patience=20,restore_best_weights=True)
	history = model.fit(inputs[train], targets[train],epochs=no_epochs,validation_data=(inputs[test], targets[test]),verbose=verbosity,callbacks=[callback,cp_callback])
	#history = model.fit(inputs[train], targets[train],epochs=no_epochs,verbose=verbosity)
	
	#save entire model
	model.save('resnet50_'+str(fold_no)+'.model')
	
	#save img of acc and loss
	fig0 = plt.figure(figsize=(12,6))
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	#plt.show()
	fig0.savefig('accuracy-'+str(fold_no)+'.png')

	fig1 = plt.figure(figsize=(12,6))
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	#plt.show()
	fig1.savefig('loss-'+str(fold_no)+'.png')
	
	# Generate generalization metrics
	scores = model.evaluate(inputs[test], targets[test], verbose=0)
	acc_per_fold.append(scores[1])
	loss_per_fold.append(scores[0])
	#calculate prediction
	value = model.predict(inputs[test])
	y_pred =np.argmax(value,axis=1)
	y_true = np.argmax(targets[test],axis=1)
	confusion = confusion_matrix(y_true,y_pred)
	np.save('conf-'+str(fold_no)+'.npy',confusion)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	#compute precision
	precision = TP / float(TP+FP)
	pric_per_fold.append(precision)
	#compute sensitivity/recall should be hihger
	sensitivity = TP / float(TP + FN)   #recall_score
	sens_per_fold.append(sensitivity)
	#compute specificity : A highly specific test can be useful for ruling in patients who have a certain disease.
	specificity = TN / float(TN + FP)
	spec_per_fold.append(specificity)
	#compute f1-score
	f1_score = (2*(precision*sensitivity))/(precision+sensitivity)
	f1_per_fold.append(f1_score)
	
	print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}')
	print(f'Score for fold {fold_no}: sensitivity of {sensitivity}; specificity of {specificity}; f1 of {f1_score}')

	# Increase fold number
	fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
	print('------------------------------------------------------------------------')
	print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)} (+- {np.std(loss_per_fold)})')
print(f'> Sensitivity: {np.mean(sens_per_fold)} (+- {np.std(sens_per_fold)})')
print(f'> precision: {np.mean(pric_per_fold)} (+- {np.std(pric_per_fold)})')
print(f'> f1-score: {np.mean(f1_per_fold)} (+- {np.std(f1_per_fold)})')
print('------------------------------------------------------------------------')

