import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import misc
img = misc.imread('./ELE.png')


img_tf = tf.Variable(img)
print(img_tf.get_shape().as_list())# print the shape of the matrix of the input image in list 


img = plt.imread('ELE.png')
plt.imshow(img)
plt.figure()# print the input image

img_red = img[:, :, 0]
plt.imshow(img_red, cmap=plt.cm.gray)
plt.figure()#print the gray of input image
img_tf = tf.Variable(img_red)
print(img_tf.get_shape().as_list())# shape of the matrix in list, of the gray version of the input image 

img_tiny = img[::6, ::6]
plt.imshow(img_tiny, interpolation='nearest')
img_tf = tf.Variable(img_tiny) # shape of the matrix in list, of tiny version of the input image
print(img_tf.get_shape().as_list())
plt.show() #print the low resolution or tiny of the input image

