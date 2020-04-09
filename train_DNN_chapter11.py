# why we need activation function: if without activation function, all layer can combine to 1 layer - not deep
# backpropagation: https://dominhhai.github.io/vi/2018/04/nn-intro/#5-lan-truy%E1%BB%81n-ng%C6%B0%E1%BB%A3c-v%C3%A0-%C4%91%E1%BA%A1o-h%C3%A0m
# Read document above before far.
#
# IN MY OPINION, JUST USE DEFAULT PARAMETER BEFORE TRY TECHNICAL BELOW, JUST USE IT WHEN YOU REALLY FACE WITH PROBLEM
#-------------------------------------------------
# Some step need to concern when training a DNN. I will list down details, just some bullet point to google it
# 1. Glorot and He Initialization: the way to random initial weight for DNN
# 2. Chosee Activation Functions. ReLU is prefer but still have variation like Noisy ReLU, Leaky ReLu, ELUs.
# And why didn't use sgn, sigmoid or tanh (https://machinelearningcoban.com/2017/02/24/mlp/#-relu)
# 3.Batch Normalization
# 4.Gradient Clipping; I don't really understand how it work
#===> all technical above to help DNN avoid The Vanishing/Exploding Gradients Problems
#------------------------------------------------
# 6.Reusing Pretrained Layers

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

def main():
	fashion_mnist = keras.datasets.fashion_mnist
	(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

	X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
	y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

	model_A = keras.models.load_model("my_keras_model.h5")
	# model_A.get_layer(name='dense').name='dense_reuse'
	model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
	model_B_on_A.add(keras.layers.Dense(10, activation="softmax", name="new_dense"))

	for layer in model_B_on_A.layers[:-1]:
		layer.trainable = False
	#fereeze two first layer	
	model_B_on_A.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",
		metrics=["accuracy"])

	history = model_B_on_A.fit(X_train, y_train, epochs=4,
		validation_data=(X_valid, y_valid))
	for layer in model_B_on_A.layers[:-1]:
		layer.trainable = True
		#unereeze two first layer
	optimizer = keras.optimizers.SGD(lr=1e-4) # the default lr is 1e-2
	model_B_on_A.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
		metrics=["accuracy"])
	history = model_B_on_A.fit(X_train, y_train, epochs=4,
		validation_data=(X_valid, y_valid))

if __name__ == '__main__':
	main()

# 7.Unsupervised Pretraining: we have a lot of unlabel data, don't waste it, use it to train unsupervise
# NN then reuse lower layer and add new layer to pretrain supervise DNN.
# 8.Pretraining on an Auxiliary Task
# ====> all 3 technical above are pretrain, DNN will work well when we have a lot of data, but you don't
# so reuse other NN network had been train before. Standing on the shoulders of giants
#------------------------------------------------
# 9.find faster optimizer
# 10.Learning rate scheduling, descending over time
#====> Make DNN faster to find optimization weight
#-----------------------------------------------
# 11.l2 and l1 Regularization: need to clear when to use each refular (p364)
