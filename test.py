import os
import cv2
import numpy as np
import pickle
from threading import Thread
from matplotlib import pyplot as plt
from keras.models import load_model

# train imagelar ile test edince doğru veriyor
path = 'images/test'

print('Model is loading')
# model = load_model('saved_weights/siamese_lr_0.007_adapted.h5')
print('Model is loaded')

train_path = 'images/train/'
test_path = 'images/test/'
pairs = [None for _ in range(2)]
list_train = []
names = []


camera_book_names = ['hayvan', 'sayitut', 'sefiller', 'sokrates', 'sultan']
for name in camera_book_names:
	if name == 'sultan' or name == 'sokrates':
		pass
	else:
		list_train.append(cv2.resize(cv2.imread(test_path + name + '_test1.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
		names.append(name + '_test1.png')

for i in range(190):
	list_train.append(cv2.resize(cv2.imread(train_path + 'book_' + str(i) + '.png'), (100, 150), interpolation=cv2.INTER_CUBIC))
	names.append('book_' + str(i) + '.png')




# for book in os.listdir(train_path):
# 	print('Loading', book)
# 	book_path = os.path.join(train_path, book)
# 	train_image = cv2.resize(cv2.imread(book_path), (100, 150), interpolation=cv2.INTER_CUBIC)
# 	list_train.append(train_image)
# 	names.append(book)

pairs[1] = np.array(list_train)


# for i in range(len(list_train)):
# 	img = list_train[i]
# 	plt.title(names[i])
# 	plt.imshow(img)
# 	plt.show()


def guess(img):
	list_test = []
	for _ in range(195):
		list_test.append(img)
	pairs[0] = np.array(list_test)
	predictions = model.predict(pairs)
	index = np.argmax(predictions)
	print(str(index) + ':', max(predictions))
	return pairs[1][index], max(predictions), index


guesses = []
if __name__ == '__main__':
	for name in os.listdir(path):
		image_path = os.path.join(path, name)
		image = cv2.resize(cv2.imread(image_path), (100, 150), interpolation=cv2.INTER_CUBIC)
		print('Guessing', name)
		result, predictmax, index = guess(image)
		guesses.append({'name': name, 'result': names[index], 'predictmax': predictmax, 'index': index})

pickle.dump(guesses, open('guesses_0.007_adapted_same_loading_as_augmented_nn.p', 'wb'))
