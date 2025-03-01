import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, ThresholdedReLU, BatchNormalization, merge, GlobalAveragePooling2D
from keras import backend as K
from keras import regularizers
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.initializers import glorot_normal

names = ['hayvan', 'sayitut', 'sefiller', 'sokrates', 'sultan']
num_test_classes = len(names)
num_train_classes = 0
global mode, model, X_train_3ch, X_test_3ch, Y_train, Y_test


def load_data():
	train_path = 'images/train/'
	test_path = 'images/test/'
	train_img = []
	test_img = []
	num_scrapped_book = 190
	num_scrapped_book = 5
	# Collect initials
	for name in names:
#		train_img.append(cv2.resize(cv2.imread(train_path + name + '1.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
		train_img.append(cv2.resize(cv2.imread(train_path + name + '2.png'), (100, 150), interpolation=cv2.INTER_CUBIC)) # ilk kitap thumbnail
		test_img.append(cv2.resize(cv2.imread(test_path + name + '_test1.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
		train_img.append(cv2.resize(cv2.imread(test_path + name + '_test2.png'), (100, 150), interpolation=cv2.INTER_CUBIC))

	first_books_test_labels = [val for val in range(num_test_classes) for _ in range(1)]

	# Collect web-scrapped books
	for i in range(num_scrapped_book):
		train_img.append(cv2.resize(cv2.imread(train_path + 'book_' + str(i) + '.png'), (100, 150), interpolation=cv2.INTER_CUBIC))

	global num_train_classes
	num_train_classes = num_test_classes + num_scrapped_book

	x = []
	for i in range(num_test_classes):
		x.append(i)
		x.append(i)

	for i in range(num_test_classes, num_test_classes + num_scrapped_book):
		x.append(i)

	return (np.array(train_img), np.array(x)), (np.array(test_img), np.array(first_books_test_labels))
	# return (np.array(train_img), np.arange(num_train_classes)), (np.array(test_img), np.array(first_books_test_labels))


def create_model1():
	m = Sequential()

	m.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 100, 3)))
	m.add(Conv2D(32, (3, 3), activation='relu'))
	m.add(MaxPooling2D(pool_size=(2, 2)))
	m.add(Dropout(0.25))

	m.add(Conv2D(32, (3, 3), activation='relu'))
	m.add(MaxPooling2D(pool_size=(2, 2)))
	m.add(Dropout(0.25))

	m.add(Flatten())
	m.add(Dense(128, activation='relu'))
	m.add(Dropout(0.5))
	m.add(Dense(num_train_classes, activation='softmax'))

	print('Model1 Summary')
	m.summary()
	exit()

	return m


def create_model2():
	m = Sequential()

	# (Conv -> Relu -> Conv -> Relu -> MaxPool) * 3 -> Flat -> Dense

	for u in range(3):
		if u == 0:
			m.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 100, 3)))
		else:
			m.add(Conv2D(32, (3, 3), activation='relu'))
		m.add(ThresholdedReLU(0))
		m.add(Conv2D(32, (3, 3), activation='relu'))
		m.add(ThresholdedReLU(0))

		m.add(MaxPooling2D(pool_size=(2, 2)))

	m.add(Flatten())
	m.add(Dense(num_train_classes, activation='softmax'))

	print('num_train_classes:', num_train_classes)
	print('Model2 Summary')
	m.summary()
	exit()

	return m


def create_model_vgg16():
	from keras.applications.vgg16 import VGG16

	input_tensor = Input(shape=(150, 100, 3))
	base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

	for layer in base_model.layers[:15]:
		layer.trainable = False

	top_model = Sequential()
	top_model.add(ZeroPadding2D((2, 2), input_shape=base_model.output_shape[1:]))
	top_model.add(Conv2D(32, (5, 5), activation='relu', kernel_regularizer=regularizers.l1(0.0001), kernel_initializer=glorot_normal()))
	# top_model.add(ZeroPadding2D((1, 1)))
	top_model.add(MaxPooling2D(pool_size=(2, 2)))
	top_model.add(Dropout(0.25))
	top_model.add(BatchNormalization())
	top_model.add(ThresholdedReLU(0))

	# top_model.add(ZeroPadding2D((2, 2)))
	# top_model.add(Conv2D(32, (5, 5), activation='relu'))
	# top_model.add(ZeroPadding2D((1, 1)))
	# top_model.add(MaxPooling2D(pool_size=(2, 2)))
	# top_model.add(Dropout(0.25))
	# top_model.add(BatchNormalization())
	# top_model.add(ThresholdedReLU(0))
	#
	# top_model.add(ZeroPadding2D((2, 2)))
	# top_model.add(Conv2D(32, (5, 5), activation='relu'))
	# top_model.add(ZeroPadding2D((1, 1)))
	# top_model.add(MaxPooling2D(pool_size=(2, 2)))
	# top_model.add(Dropout(0.25))
	# top_model.add(BatchNormalization())
	# top_model.add(ThresholdedReLU(0))

	# top_model.add(MaxPooling2D(pool_size=(2, 2)))

	# top_model.add(Flatten())
	top_model.add(GlobalAveragePooling2D())
	top_model.add(BatchNormalization())
	top_model.add(Dense(num_train_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.0001), kernel_initializer=glorot_normal()))
	# top_model.add(Dropout(0.2))

	m = Model(inputs=base_model.input, outputs=top_model(base_model.output))

	print('Model3 Summary')
	m.summary()
	exit()

	return m


def create_train_generator(datagen, batch_size=32):
	X = X_train_3ch
	cls_num = X.shape[0]
	batch_size = min(batch_size, cls_num - 1)
	pairs = [np.zeros((batch_size, 150, 100, 3)) for i in range(2)]
	targets = np.zeros((batch_size,))
	targets[batch_size//2:] = 1

	while True:
		categories = np.random.choice(cls_num, size=(batch_size,), replace=False)
		for i in range(batch_size):
			category = categories[i]
			pairs[0][i, :, :, :] = datagen.random_transform(X[category])
			category_2 = category if i >= batch_size // 2 else (category + np.random.randint(1, cls_num)) % cls_num
			pairs[1][i, :, :, :] = datagen.random_transform(X[category_2])
		yield (pairs, targets)


def create_test_generator(datagen=None, batch_size=32):
	pairs = [np.zeros((batch_size, 150, 100, 3)) for i in range(2)]
	targets = np.zeros((batch_size,))
	# targets[batch_size//2:] = 1
	targets[:] = 1

	while True:
		for i in range(batch_size):
			category_test = np.random.randint(0, X_test_3ch.shape[0])
			category_train = category_test // 2
			pairs[0][i, :, :, :] = X_test_3ch[category_test]
			pairs[1][i, :, :, :] = X_train_3ch[category_train]
		yield (pairs, targets)


def augmentation_fit():
	global mode
	mode = 'augmentation'
	datagen = ImageDataGenerator(
		rotation_range=15,
		width_shift_range=0.05,
		height_shift_range=0.05,
		shear_range=0.05,
		zoom_range=0.05,
		horizontal_flip=True,
		# TODO fill_mode='constant')  # Constant zero
		fill_mode='constant')  # Constant zero

	datagen.fit(X_train_3ch)
	train_generator = create_train_generator(datagen)
	test_generator = create_test_generator(datagen, batch_size=4)

	# for (pairs, targets) in train_generator:
	# 	for i in range(len(targets)):
	# 		img1, img2 = pairs[0][i, :, :, ::-1], pairs[1][i, :, :, ::-1]
	# 		result = 'Same' if targets[i] == 1 else 'Different'
	# 		plt.suptitle(result)
	# 		plt.subplot(1, 2, 1)
	# 		plt.imshow(img1)
	# 		plt.subplot(1, 2, 2)
	# 		plt.imshow(img2)
	# 		plt.show()

	# TODO steps_per_epoch=20, epochs=200
	return model.fit_generator(datagen.flow(X_train_3ch, Y_train, batch_size=32), steps_per_epoch=10, epochs=30, validation_data=(X_test_3ch, Y_test))#, validation_data=test_generator, validation_steps=3)


def normal_fit():
	global mode
	mode = 'normal'
	return model.fit(X_train_3ch, Y_train, batch_size=32, epochs=600, validation_data=(X_test_3ch, Y_test))


def siamese(smodel):
	input_shape = (150, 100, 3)
	left_input = Input(input_shape)
	right_input = Input(input_shape)

	encoded_l = smodel(left_input)
	encoded_r = smodel(right_input)

	L1 = lambda x: K.abs(x[0] - x[1])
	both = merge([encoded_l, encoded_r], mode=L1, output_shape=lambda x: x[0])

	prediction = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.0001), kernel_initializer=glorot_normal())(both)

	return Model(inputs=[left_input, right_input], outputs=prediction)


def run(lr=0.001, augmented=True, modelno=3, optimizer='sgd'):  # If modelno changes, change the model_name (vgg16 part)
	global model, X_train_3ch, X_test_3ch, Y_train, Y_test
	# Load images
	(X_train, y_train), (X_test, y_test) = load_data()

	print(len(X_train), len(y_train), len(X_test), len(y_test))
	# exit()

	# Adjust sizes
	Y_train = np_utils.to_categorical(y_train, num_train_classes)
	Y_test = np_utils.to_categorical(y_test, num_train_classes)

	X_train_3ch = X_train.reshape(X_train.shape[0], 150, 100, 3)
	X_test_3ch = X_test.reshape(X_test.shape[0], 150, 100, 3)

	X_train_3ch = X_train_3ch.astype('float32') / 255
	X_test_3ch = X_test_3ch.astype('float32') / 255

	if modelno == 1:
		model = create_model1()
	elif modelno == 2:
		model = create_model2()
	else:  # default
		model = create_model_vgg16()

	# model = siamese(model)

	if optimizer == 'sgd':
		opt = SGD(lr=lr, decay=1e-4, momentum=0.99, nesterov=True)
	else:
		opt = Adam(0.00006)
	# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	# model.count_params()

	print('learning rate is', K.get_value(model.optimizer.lr))

	if augmented:
		history = augmentation_fit()
	else:
		history = normal_fit()

	# model_name = 'siamese_lr_' + str(lr) + 'small_augmentation'
	model_name = str(len(X_train)) + 'x' + str(len(X_test)) + '_' + K.backend()
	model_name += '_augmenatation' if augmented else '_normal'
	model_name += '_lr' + str(lr) + '_model' + str(modelno)
	print('model_name:', model_name)
	pickle.dump(history.history, open('final_report/histories/' + model_name + '.p', 'wb'))

	# model.save('saved_weights/' + model_name + '.h5')


# run(lr=0.01, optimizer='sgd', modelno=3, augmented=True)
run(lr=0.01, optimizer='sgd', modelno=3, augmented=False)
















# import cv2
# import numpy as np
# import pickle
# from matplotlib import pyplot as plt
# from keras.models import Sequential, Model
# from keras.layers.core import Dense, Dropout, Flatten, Dropout
# from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, ThresholdedReLU, BatchNormalization, merge, GlobalAveragePooling2D
# from keras import backend as K
# from keras import regularizers
# from keras.utils import np_utils
# from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import SGD, Adam
# from keras.initializers import glorot_normal
#
# names = ['hayvan', 'sayitut', 'sefiller', 'sokrates', 'sultan']
# num_test_classes = len(names)
# num_train_classes = 0
# global mode, model, X_train_3ch, X_test_3ch, Y_train, Y_test
#
#
# def load_data():
# 	train_path = 'images/train/'
# 	test_path = 'images/test/'
# 	train_img = []
# 	test_img = []
# 	num_scrapped_book = 190
# 	# Collect initials
# 	for name in names:
# 		train_img.append(cv2.resize(cv2.imread(train_path + name + '2.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
# 		# Doing for adaptation
# 		# put some of the test images into train images
# 		if name == 'sultan' or name == 'sokrates':
# 			test_img.append(cv2.resize(cv2.imread(test_path + name + '_test1.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
# 		else:
# 			train_img.append(cv2.resize(cv2.imread(test_path + name + '_test1.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
# 		test_img.append(cv2.resize(cv2.imread(test_path + name + '_test2.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
#
# 	# first_books_test_labels = [val for val in range(num_test_classes) for _ in range(2)]
# 	first_books_test_labels = [0, 1, 2, 3, 3, 4, 4]
#
# 	# Collect web-scrapped books
# 	for i in range(num_scrapped_book):
# 		train_img.append(cv2.resize(cv2.imread(train_path + 'book_' + str(i) + '.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
#
# 	global num_train_classes
# 	num_train_classes = num_test_classes + num_scrapped_book
#
# 	first_books_train_labels = [0, 0, 1, 1, 2, 2, 3, 4]
# 	train_labels = [x for x in range(num_test_classes, num_test_classes + num_scrapped_book)]
#
# 	return (np.array(train_img), np.array(first_books_train_labels + train_labels)),\
# 		   (np.array(test_img), np.array(first_books_test_labels))
#
#
# def create_model1():
# 	m = Sequential()
#
# 	m.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 100, 3)))
# 	m.add(Conv2D(32, (3, 3), activation='relu'))
# 	m.add(MaxPooling2D(pool_size=(2, 2)))
# 	m.add(Dropout(0.25))
#
# 	m.add(Conv2D(32, (3, 3), activation='relu'))
# 	m.add(MaxPooling2D(pool_size=(2, 2)))
# 	m.add(Dropout(0.25))
#
# 	m.add(Flatten())
# 	m.add(Dense(128, activation='relu'))
# 	m.add(Dropout(0.5))
# 	m.add(Dense(num_train_classes, activation='softmax'))
#
# 	return m
#
#
# def create_model2():
# 	m = Sequential()
#
# 	# (Conv -> Relu -> Conv -> Relu -> MaxPool) * 3 -> Flat -> Dense
#
# 	for u in range(3):
# 		if u == 0:
# 			m.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 100, 3)))
# 		else:
# 			m.add(Conv2D(32, (3, 3), activation='relu'))
# 		m.add(ThresholdedReLU(0))
# 		m.add(Conv2D(32, (3, 3), activation='relu'))
# 		m.add(ThresholdedReLU(0))
#
# 		m.add(MaxPooling2D(pool_size=(2, 2)))
#
# 	m.add(Flatten())
# 	m.add(Dense(num_train_classes, activation='softmax'))
#
# 	return m
#
#
# def create_model_vgg16():
# 	from keras.applications.vgg16 import VGG16
#
# 	input_tensor = Input(shape=(150, 100, 3))
# 	base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
#
# 	for layer in base_model.layers[:15]:
# 		layer.trainable = False
#
# 	top_model = Sequential()
# 	top_model.add(ZeroPadding2D((2, 2), input_shape=base_model.output_shape[1:]))
# 	top_model.add(Conv2D(32, (5, 5), activation='relu', kernel_regularizer=regularizers.l1(0.0001), kernel_initializer=glorot_normal()))
# 	# top_model.add(ZeroPadding2D((1, 1)))
# 	top_model.add(MaxPooling2D(pool_size=(2, 2)))
# 	top_model.add(Dropout(0.25))
# 	top_model.add(BatchNormalization())
# 	top_model.add(ThresholdedReLU(0))
#
# 	# top_model.add(ZeroPadding2D((2, 2)))
# 	# top_model.add(Conv2D(32, (5, 5), activation='relu'))
# 	# top_model.add(ZeroPadding2D((1, 1)))
# 	# top_model.add(MaxPooling2D(pool_size=(2, 2)))
# 	# top_model.add(Dropout(0.25))
# 	# top_model.add(BatchNormalization())
# 	# top_model.add(ThresholdedReLU(0))
# 	#
# 	# top_model.add(ZeroPadding2D((2, 2)))
# 	# top_model.add(Conv2D(32, (5, 5), activation='relu'))
# 	# top_model.add(ZeroPadding2D((1, 1)))
# 	# top_model.add(MaxPooling2D(pool_size=(2, 2)))
# 	# top_model.add(Dropout(0.25))
# 	# top_model.add(BatchNormalization())
# 	# top_model.add(ThresholdedReLU(0))
#
# 	# top_model.add(MaxPooling2D(pool_size=(2, 2)))
#
# 	# top_model.add(Flatten())
# 	top_model.add(GlobalAveragePooling2D())
# 	top_model.add(BatchNormalization())
# 	top_model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.0001), kernel_initializer=glorot_normal()))
# 	top_model.add(Dropout(0.2))
#
# 	m = Model(inputs=base_model.input, outputs=top_model(base_model.output))
#
# 	return m
#
#
# def same_train(category):
# 	if category < 6:
# 		return (category // 2) * 2 + np.random.randint(2)
# 	return category
#
#
# # def same_test(category):
# # 	if category > 2:
# # 		category += 1
# # 		return (category // 2) * 2 + np.random.randint(2) - 1
# # 	return category
#
#
# def same_test(category):
# 	if category > 2:
# 		category += 1
# 		return category // 2 + 4
# 	return np.random.randint(2) + category * 2
#
#
# def different_train(category):
# 	num_train_images = X_train_3ch.shape[0]
# 	if category < 6:
# 		category = (category // 2) * 2
# 		return (category + np.random.randint(2, num_train_images)) % num_train_images
# 	return (category + np.random.randint(1, num_train_images)) % num_train_images
#
#
# def different_test(category):
# 	num_train_images = X_train_3ch.shape[0]
# 	if category > 2:
# 		category *= 2
# 		return (category + np.random.randint(2, num_train_images)) % num_train_images
# 	category += 1
# 	category = category // 2 + 4
# 	return (category + np.random.randint(1, num_train_images)) % num_train_images
#
#
# def create_train_generator(datagen, batch_size=32):
# 	X = X_train_3ch
# 	cls_num = num_train_classes
# 	batch_size = min(batch_size, cls_num - 1)
# 	pairs = [np.zeros((batch_size, 150, 100, 3)) for i in range(2)]
# 	targets = np.zeros((batch_size,))
# 	targets[batch_size//2:] = 1
#
# 	while True:
# 		categories = np.random.choice(cls_num, size=(batch_size,), replace=False)
# 		for i in range(batch_size):
# 			category = categories[i]
# 			pairs[0][i, :, :, :] = datagen.random_transform(X[category])
# 			category_2 = same_train(category) if i >= batch_size // 2 else different_train(category)
# 			pairs[1][i, :, :, :] = datagen.random_transform(X[category_2])
# 		yield (pairs, targets)
#
#
# def create_test_generator(datagen=None, batch_size=32):
# 	pairs = [np.zeros((batch_size, 150, 100, 3)) for i in range(2)]
# 	targets = np.zeros((batch_size,))
# 	targets[batch_size//2:] = 1
#
# 	while True:
# 		for i in range(batch_size):
# 			category_test = np.random.randint(0, X_test_3ch.shape[0])
# 			category_train = same_test(category_test) if i >= batch_size // 2 else different_test(category_test)
# 			pairs[0][i, :, :, :] = X_test_3ch[category_test]
# 			pairs[1][i, :, :, :] = X_train_3ch[category_train]
# 		yield (pairs, targets)
#
#
# def augmentation_fit():
# 	global mode
# 	mode = 'augmentation'
# 	datagen = ImageDataGenerator(
# 		rotation_range=15,
# 		width_shift_range=0.05,
# 		height_shift_range=0.05,
# 		shear_range=0.05,
# 		zoom_range=0.05,
# 		# horizontal_flip=True,
# 		featurewise_center=True,
# 		# TODO fill_mode='constant')  # Constant zero
# 		fill_mode='constant')  # Constant zero
#
# 	datagen.fit(X_train_3ch)
# 	train_generator = create_train_generator(datagen)
# 	test_generator = create_test_generator(datagen)
#
# 	# for (pairs, targets) in test_generator:
# 	# 	print('started new one')
# 	# 	for i in range(len(targets)):
# 	# 		img1, img2 = pairs[0][i, :, :, ::-1], pairs[1][i, :, :, ::-1]
# 	# 		result = 'Same' if targets[i] == 1 else 'Different'
# 	# 		plt.suptitle(result)
# 	# 		plt.subplot(1, 2, 1)
# 	# 		plt.title('Test')
# 	# 		plt.imshow(img1)
# 	# 		plt.subplot(1, 2, 2)
# 	# 		plt.title('Train')
# 	# 		plt.imshow(img2)
# 	# 		plt.show()
#
# 	# TODO steps_per_epoch=20, epochs=200
# 	return model.fit_generator(train_generator, steps_per_epoch=20, epochs=300, validation_data=test_generator, validation_steps=5)
#
#
# def normal_fit():
# 	global mode
# 	mode = 'normal'
# 	return model.fit(X_train_3ch, Y_train, batch_size=32, epochs=600, validation_data=(X_test_3ch, Y_test))
#
#
# def siamese(smodel):
# 	input_shape = (150, 100, 3)
# 	left_input = Input(input_shape)
# 	right_input = Input(input_shape)
#
# 	encoded_l = smodel(left_input)
# 	encoded_r = smodel(right_input)
#
# 	L1 = lambda x: K.abs(x[0] - x[1])
# 	both = merge([encoded_l, encoded_r], mode=L1, output_shape=lambda x: x[0])
#
# 	prediction = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.0001), kernel_initializer=glorot_normal())(both)
#
# 	return Model(inputs=[left_input, right_input], outputs=prediction)
#
#
# def run(lr=0.001, augmented=True, modelno=3, optimizer='sgd'):  # If modelno changes, change the model_name (vgg16 part)
# 	global model, X_train_3ch, X_test_3ch, Y_train, Y_test
# 	# Load images
# 	(X_train, y_train), (X_test, y_test) = load_data()
#
# 	print(len(X_train), len(y_train), len(X_test), len(y_test))
# 	exit()
#
# 	# TODO image'ları, labelları title'a koyarak kontrol et
# 	# X = X_test
# 	# y = y_test
# 	# X = X_train
# 	# y = y_train
# 	# for i in range(len(X)):
# 	# 	try:
# 	# 		plt.title(names[y[i]])
# 	# 	except IndexError:
# 	# 		plt.title('book_' + str(y[i]))
# 	# 	plt.imshow(X[i])
# 	# 	plt.show()
#
# 	# Adjust sizes
# 	Y_train = np_utils.to_categorical(y_train, num_train_classes)
# 	Y_test = np_utils.to_categorical(y_test, num_train_classes)
#
# 	X_train_3ch = X_train.reshape(X_train.shape[0], 150, 100, 3)
# 	X_test_3ch = X_test.reshape(X_test.shape[0], 150, 100, 3)
#
# 	X_train_3ch = X_train_3ch.astype('float32') / 255
# 	X_test_3ch = X_test_3ch.astype('float32') / 255
#
# 	if modelno == 1:
# 		model = create_model1()
# 	elif modelno == 2:
# 		model = create_model2()
# 	else:  # default
# 		model = create_model_vgg16()
#
# 	model = siamese(model)
#
# 	if optimizer == 'sgd':
# 		opt = SGD(lr=lr, decay=1e-4, momentum=0.99, nesterov=True)
# 	else:
# 		opt = Adam(0.00006)
# 	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#
# 	model.count_params()
#
# 	print('learning rate is', K.get_value(model.optimizer.lr))
#
# 	if augmented:
# 		history = augmentation_fit()
# 	else:
# 		history = normal_fit()
#
# 	model_name = 'siamese_lr_' + str(lr) + '_adapted'
# 	pickle.dump(history.history, open('siamese_histories/final_test/' + model_name + '.p', 'wb'))
#
# 	model.save('saved_weights/' + model_name + '.h5')
#
#
# run(lr=0.01, optimizer='sgd')
