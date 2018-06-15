
import tensorflow as tf
print(tf.__version__)
#1.5.0
random_int_var = tf.get_variable("random_int_var_1_to_20",
                                     initializer=tf.random_uniform([3, 3],
                                                 minval=1,
                                                 maxval=20,
                                                 dtype=tf.int32))
print(random_int_var)
#<tf.Variable 'random_int_var_1_to_20:0' shape=(3, 3) dtype=int32_ref>
tf_int_ones = tf.ones(shape=[3,3], dtype="int32")
print(tf_int_ones)
#Tensor("ones:0", shape=(3, 3), dtype=int32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(random_int_var))
#random variables from 1 to 20 with maximum value 20
print(sess.run(tf_int_ones))
#[[1 1 1]
 #[1 1 1]
 #[1 1 1]]
tf_matrix_multiplication_prod = tf.matmul(random_int_var, tf_int_ones)
print(sess.run(tf_matrix_multiplication_prod))
#multiplied the two generated matrices 

