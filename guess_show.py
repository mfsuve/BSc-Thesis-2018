import pickle
from matplotlib import pyplot as plt
import cv2
import os


# guesses = pickle.load(open('guesses_on_train.p', 'rb'))
guesses = pickle.load(open('guesses.p', 'rb'))
train_path = 'images/train'
# test_path = 'images/train'
test_path = 'images/test'

for guess in guesses:
	name = guess['name']
	result_name = guess['result']
	predictmax = guess['predictmax']
	index = guess['index']
	book = cv2.imread(os.path.join(test_path, name))
	result = cv2.imread(os.path.join(train_path, result_name))
	plt.suptitle(name + '\nmax prediction: ' + str(predictmax) + '\nindex: ' + str(index))
	plt.subplot(1, 2, 1)
	plt.imshow(book[:, :, ::-1])
	plt.subplot(1, 2, 2)
	plt.imshow(result[:, :, ::-1])
	plt.show()
