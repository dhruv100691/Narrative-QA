import tensorflow as tf
import numpy as np

mat_a = tf.constant(np.arange(1,12, dtype=np.int32), shape=[2,2,1,3],dtype='float')
mat_b = tf.constant(np.arange(12,21, dtype=np.int32), shape=[1,1,3,3],dtype='float')
mat_e = tf.constant(np.arange(12,21, dtype=np.int32), shape=[3,3],dtype='float')
mat_f = tf.fill([1,3], 2.0, name='PAD')
mat_g = tf.constant(np.arange(1,6, dtype=np.int32), shape=[2,1,1,3],dtype='float')
mat_l= tf.constant(np.arange(1,5, dtype=np.int32),shape=[2,2,1,1], name='PAD',dtype='float')
mat_i = tf.constant(np.arange(1,12, dtype=np.int32), shape=[2,2,3],dtype='float')
mat_m = tf.constant(np.arange(1,10, dtype=np.int32), shape=[3,3],dtype='float')

mat_c = tf.nn.conv2d(mat_a,mat_b,[1,1,1,1],"SAME")
mat_a1 = tf.reshape(mat_a,[-1,3])
#mul_c = tf.matmul(mat_a, mat_b)
mul_c2 = tf.matmul(mat_a1,mat_e)
mul_c2 = tf.reshape(mul_c2,[2,2,1,3])
mul_c3 = mat_l * mul_c2
print ("shape",mul_c3.get_shape().as_list())
#mat_d = tf.expand_dims(mat_a,2)

with tf.Session() as sess:
   runop,a ,b,runop2,runop3,runop4,g,l ,i,m,op_5= sess.run([mat_c,mat_a,mat_b,mul_c2,mul_c3,mat_a+mat_g,mat_g,mat_l,mat_i,mat_m,tf.matmul(mat_i,mat_m)])

print ("A",a)
print ("G",g)
print ("sum",runop4)
print ("B",b)
print ("product",runop)
print ("new product",runop2)
print ("modified prod",runop3)
print ("L",l)
print ("I",i)
print ("M",m)
print ("new op",op_5)

