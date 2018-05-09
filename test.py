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
model = load_model('saved_weights/siamese_lr_0.01_two_blocks.h5')
print('Model is loaded')

train_path = 'images/train'
# pairs = [np.zeros((195, 150, 100, 3)) for _ in range(2)]
pairs = [None for _ in range(2)]
list_train = []
names = []
for book in os.listdir(train_path):
	print('Loading', book)
	book_path = os.path.join(train_path, book)
	train_image = cv2.resize(cv2.imread(book_path), (100, 150), interpolation=cv2.INTER_CUBIC)
	list_train.append(train_image)
	names.append(book)

pairs[1] = np.array(list_train)


def guess(img):
	list_test = []
	for _ in range(195):
		list_test.append(img)
	pairs[0] = np.array(list_test)
	predictions = model.predict(pairs, verbose=1)
	print('predictions:', predictions)
	index = np.argmax(predictions)
	return pairs[1][index], max(predictions), index


guesses = []
if __name__ == '__main__':
	for name in os.listdir(path):
		image_path = os.path.join(path, name)
		image = cv2.resize(cv2.imread(image_path), (100, 150), interpolation=cv2.INTER_CUBIC)
		print('Guessing', name)
		result, predictmax, index = guess(image)
		guesses.append({'name': name, 'result': names[index], 'predictmax': predictmax, 'index': index})
		# plt.title(name + '\nmax prediction: ' + str(predictmax), result)
		# plt.imshow(result[:, :, ::-1])
		# plt.show()

pickle.dump(guesses, open('guesses.p', 'wb'))
