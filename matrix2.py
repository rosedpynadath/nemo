import scipy
import tensorflow


import scipy.misc
from scipy import misc
import tensorflow as tf

sess = tf.InteractiveSession()#The only difference between Session and an 
#InteractiveSession is that InteractiveSession makes itself the default session so that you can call run() or eval() without explicitly calling the session.
#because it avoids having to pass an explicit Session object to run operations.sess.run(m1)

matrix1 = tf.constant([[3., 3.]])# 3. means 3.0,   Tensor("Const:0", shape=(1, 2), dtype=float32)
matrix2 = tf.constant([[2.],[2.]])#Tensor("Const_1:0", shape=(2, 1), dtype=float32)
product = tf.matmul(matrix1, matrix2)
#print the product
print(product.eval())#[[12.]]

#close the session to release resources-------------------------------------***----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

sess.close()
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)
init = tf.global_variables_initializer()#eturns an op that initializes all variables.it's updates version is intialze global variables.other one is deprecated.
#array([[12.]], datatype-float32
sess = tf.Session()

sess.run(init)

print(sess.run([product]))
#close the session-----------------------------------------------------------***----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
sess = tf.InteractiveSession() # see the answers above :)
x = [[1.,2.,1.],[1.,1.,1.]]    # a 2D matrix as input to softmax, [[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]], x matrix
y = tf.nn.softmax(x)           # this is the softmax function,Tensor("Softmax:0", shape=(2, 3), dtype=float32),The softmax function is often used in the final layer of a neural network-based classifier.
                               #In building neural networks softmax functions used in different layer level.
                               # we can put any function to y
u = y.eval()
print(u) #[[0.21194157 0.5761169  0.21194157]
#          [0.33333334 0.33333334 0.33333334]] u matrix with shape = [2,3]  2 rows, 3 columns matrix






