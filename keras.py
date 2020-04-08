from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


def main(type=0):
	'''
	Type = 0 is run model without saving
	type = 1 is run model with callback checkpoint
	type = 2 is run model with earlystop
	type = 3 is run and save tensorboard
	'''
	fashion_mnist = keras.datasets.fashion_mnist
	(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

	#split train set to train and valid set
	#dividing 225 to to convert pixel data to 0-1 range
	X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
	y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

	class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
				"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

	model = keras.models.Sequential()
	model.add(keras.layers.Flatten(input_shape=[28, 28]))
	model.add(keras.layers.Dense(300, activation="relu"))
	model.add(keras.layers.Dense(100, activation="relu"))
	model.add(keras.layers.Dense(10, activation="softmax"))

	#flatten to convert input matrix 28x28 to 1D array before feed to layer

	model.compile(loss="sparse_categorical_crossentropy",
				optimizer="sgd",
				metrics=["accuracy"])

	# notice loss function, some function are: sparse_categorical_crossentropy
	# categorical_crossentropy, binary_crossentropy depend on output data.
	# can use class_weight when fit model to pretend very skewed data (taget_data).
	# similar is sample_weight with sample data
	# validation_split

#----------------------------------------------------------------------------------------------
	checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
	root_logdir = os.path.join(os.curdir, "output")

	def get_run_logdir():
		import time
		run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
		return os.path.join(root_logdir, run_id)

	if type == 0:
		history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))
		# in this case i want to save model:
		# model.save("my_keras_model.h5")
		# model = keras.models.load_model("my_keras_model.h5")
	elif type == 1:
		# CHECKPOINT
		history = model.fit(X_train, y_train, epochs=5,validation_data=(X_valid, y_valid)
			,callbacks=[checkpoint_cb])
		model = keras.models.load_model("my_keras_model.h5")
	elif type ==3:
		#EARLY STOPPING
		early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
			restore_best_weights=True)
		history = model.fit(X_train, y_train, epochs=100,
			validation_data=(X_valid, y_valid),
			callbacks=[checkpoint_cb, early_stopping_cb])
	else:
		run_logdir = get_run_logdir()
		tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
		history = model.fit(X_train, y_train, epochs=5,
			validation_data=(X_valid, y_valid),
			callbacks=[tensorboard_cb])

#-----------------------------------------------------------------------------------------------
	# vizualy historical data 
	pd.DataFrame(history.history).plot(figsize=(8, 5))
	plt.grid(True)
	plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
	plt.show()

	#predict
	X_new = X_test[:3]
	y_pred = model.predict_classes(X_new)
	y_new = y_test[:3]
	print("y_pred:", y_pred, "y:", y_new)


if __name__ == '__main__':
	input_value = sys.argv[1]
	main(input_value)

# more loss function in keras, check it out:https://keras.io/losses/
# optimizer: https://keras.io/optimizers/
# another highlight is Regularization: https://labs.septeni-technology.jp/technote/ml-10-regularization-overfitting-and-underfitting/
# lossfunction vs costfunction: https://stats.stackexchange.com/questions/179026/objective-function-cost-function-loss-function-are-they-the-same-thing
#-----------------------------------------------------------------
#EXTRA MODEL: wide&deep network
class WideAndDeepModel(keras.Model):
	def __init__(self, units=30, activation="relu", **kwargs):
		super().__init__(**kwargs) # handles standard args (e.g., name)
		self.hidden1 = keras.layers.Dense(units, activation=activation)
		self.hidden2 = keras.layers.Dense(units, activation=activation)
		self.main_output = keras.layers.Dense(1)
		self.aux_output = keras.layers.Dense(1)
	def call(self, inputs):
		input_A, input_B = inputs
		hidden1 = self.hidden1(input_B)
		hidden2 = self.hidden2(hidden1)
		concat = keras.layers.concatenate([input_A, hidden2])
		main_output = self.main_output(concat)
		aux_output = self.aux_output(hidden2)
		return main_output, aux_output

# compare add and concatenate layer: https://stats.stackexchange.com/questions/361018/when-to-add-layers-and-when-to-concatenate-in-neural-networks
# to use grid search in keras: details in p320

#???: how to choose optimizier, loss function, number of layer, number of node, activation function, p325
