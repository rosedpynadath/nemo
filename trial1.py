import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import misc
img = misc.imread('./thanos.jpeg')


img_tf = tf.Variable(img)
print(img_tf.get_shape().as_list())# print the shape of the matrix of the thanos in list 


img = plt.imread('thanos.jpeg')
plt.imshow(img)
plt.figure()# print the input image

img_red = img[:, :, 0]
plt.imshow(img_red, cmap=plt.cm.gray)
plt.figure()#print thanos in grey
img_tf = tf.Variable(img_red)
print(img_tf.get_shape().as_list())# shape of the matrix in list, of the gray  thanos 

img_tiny = img[::6, ::6]
plt.imshow(img_tiny, interpolation='nearest')
img_tf = tf.Variable(img_tiny) # shape of the matrix in list, of tiny version of thanos
print(img_tf.get_shape().as_list())
plt.show() #print the low resolution or tiny of thanos

