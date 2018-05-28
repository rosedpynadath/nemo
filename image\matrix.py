import tensorflow as tf
import matplotlib.pyplot as plt #define the name plt
from scipy import misc
img = misc.imread('/Users/rosedpynadath/Desktop/thanos.jpeg')


img_tf = tf.Variable(img)          # define a variable to hold image
print(img_tf.get_shape().as_list())# print the shape of the matrix of the thanos in list 
img = plt.imread('thanos.jpeg')
plt.imshow(img)
plt.figure()                       # print command for the input image

#**********************************************

img_red = img[:, :, 0] #varrying the picture information by accessing the image matrix and obtaing gray image of the input image
plt.imshow(img_red, cmap=plt.cm.gray)# varying the image RGB scale to obtain gray image.
plt.figure()#print thanos in grey
img_tf = tf.Variable(img_red)
print(img_tf.get_shape().as_list())# shape of the matrix in list, of the gray  thanos 

#***********************************************
img_tiny = img[::6, ::6]                     # accessing the picture matix
plt.imshow(img_tiny, interpolation='nearest')#interpolation = ‘none’ works well when a big image is scaled down, 
                                             #while interpolation = ‘nearest’ works well when a small image is scaled up
img_tf = tf.Variable(img_tiny) 
                             # shape of the matrix in list, of tiny version of thanos
print(img_tf.get_shape().as_list())
plt.show()                   #created the low resolution or tiny of thanos

