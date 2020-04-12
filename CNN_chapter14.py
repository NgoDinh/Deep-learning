# Good introduction to CNN besides reading book: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
import os, shutil
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def make_data_set():
	original_dataset_dir = '/home/vu/data_science/deeplearning/dataset/dogs-vs-cats/train'
	base_dir = '/home/vu/data_science/deeplearning/dataset/catanddog'
	os.mkdir(base_dir)

	train_dir = os.path.join(base_dir, 'train')
	os.mkdir(train_dir)
	validation_dir = os.path.join(base_dir, 'validation')
	os.mkdir(validation_dir)
	test_dir = os.path.join(base_dir, 'test')
	os.mkdir(test_dir)

	train_cats_dir = os.path.join(train_dir, 'cats')
	os.mkdir(train_cats_dir)

	train_dogs_dir = os.path.join(train_dir, 'dogs')
	os.mkdir(train_dogs_dir)

	validation_cats_dir = os.path.join(validation_dir, 'cats')
	os.mkdir(validation_cats_dir)

	validation_dogs_dir = os.path.join(validation_dir, 'dogs')
	os.mkdir(validation_dogs_dir)

	test_cats_dir = os.path.join(test_dir, 'cats')
	os.mkdir(test_cats_dir)

	test_dogs_dir = os.path.join(test_dir, 'dogs')
	os.mkdir(test_dogs_dir)

	fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
	for fname in fnames:
	    src = os.path.join(original_dataset_dir, fname)
	    dst = os.path.join(train_cats_dir, fname)
	    shutil.copyfile(src, dst)

	fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
	for fname in fnames:
	    src = os.path.join(original_dataset_dir, fname)
	    dst = os.path.join(validation_cats_dir, fname)
	    shutil.copyfile(src, dst)

	fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
	for fname in fnames:
	    src = os.path.join(original_dataset_dir, fname)
	    dst = os.path.join(test_cats_dir, fname)
	    shutil.copyfile(src, dst)

	fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
	for fname in fnames:
	    src = os.path.join(original_dataset_dir, fname)
	    dst = os.path.join(train_dogs_dir, fname)
	    shutil.copyfile(src, dst)

	fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
	for fname in fnames:
	    src = os.path.join(original_dataset_dir, fname)
	    dst = os.path.join(validation_dogs_dir, fname)
	    shutil.copyfile(src, dst)

	fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
	for fname in fnames:
	    src = os.path.join(original_dataset_dir, fname)
	    dst = os.path.join(test_dogs_dir, fname)
	    shutil.copyfile(src, dst)

# before we go futher when using CNN to classify dog and and cat we need to clear some important point
# Besides a basic structure of DNN we have use in chapter 11
# 1: why we need filter in CNN: https://towardsdatascience.com/a-beginners-guide-to-convolutional-neural-networks-cnns-14649dbddce8
# 2: Role of Pooling layer in CNN.
# 3: How to choose: right parameter for filter and pooling - I need work more to findout, google have it
# but i think we need deep understand before change default selection.
def build_model():
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu',
	                        input_shape=(150, 150, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(128, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(128, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Flatten())
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.summary()
	return model

def run_model():

	# first we need to convert image data to pixel
	train_dir = '/home/vu/data_science/deeplearning/dataset/catanddog/train'
	validation_dir = '/home/vu/data_science/deeplearning/dataset/catanddog/validation'
	train_datagen = ImageDataGenerator(rescale=1./255)
	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
	        train_dir,
	        target_size=(150, 150),
	        batch_size=20,
	        class_mode='binary') #notice to fill class_model, in this situation we just have two class.

	validation_generator = test_datagen.flow_from_directory(
	        validation_dir,
	        target_size=(150, 150),
	        batch_size=20,
	        class_mode='binary')

	model = build_model()
	model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
	history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
	#we have 30 epochs to train model, with epoch we will feed 100*20 image where 100 is step_per_epoch and 20 is batch_size
	model.save('cats_and_dogs_small_chapter14.h5')
	hist_df = pd.DataFrame(history.history) 
	hist_csv_file = 'cats_and_dogs_small_chapter14_history.csv'
	with open(hist_csv_file, mode='w') as f:
	    hist_df.to_csv(f)

def view_history(file_name):

	history = pd.read_csv(file_name)
	acc = history['acc']
	val_acc = history['val_acc']
	loss = history['loss']
	val_loss = history['val_loss']

	epochs = range(len(acc))

	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()

	plt.figure()

	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()

	plt.show()



def main(type=1, file_name=None):
	if type == 0:
		make_data_set()
	elif type == 1:
		build_model()
	elif type == 2:
		view_history(file_name)
	else:
		run_model()

if __name__ == '__main__':
	input_value = int(sys.argv[1])
	file_name = sys.argv[2]
	main(input_value, file_name)

# We can see that model had been overfit, so to avoid this we use two technique is: data augmentation
# and dropout.
