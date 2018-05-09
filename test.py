import os
import cv2
from matplotlib import pyplot as plt
from keras.models import load_model

path = 'images/test'


model = load_model('saved_weights/siamese_lr_0.003.h5')
def guess(img, batch_size=32):
	train_path = 'images/train'
	pairs = [np.zeros((batch_size, 150, 100, 3)) for _ in range(2)]
	i = 0
	for book in os.listdir(train_path):
		book_path = os.path.join(book, train_path)
		train_image = cv2.imread(book_path)
		pairs[0][i, :, :, :] = img
		pairs[1][i, :, :, :] = train_image
		i += 1
		if i > batch_size:
			i = 0
			predictions = model.predict(pairs)
			print('predictions:', predictions)


if __name__ == '__main__':
	for name in os.listdir(path):
		image_path = os.path.join(name, path)
		image = cv2.imread(image_path)
		result = guess(image)
		plt.imshow(result[:, :, ::-1])
		plt.show()
