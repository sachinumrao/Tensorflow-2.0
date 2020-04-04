import tensorflow as tf
print(tf.__version__)

# Tensor datatypes
st1 = tf.Variable("This is Sachin", tf.string)
print(st1)

num1 = tf.Variable(32, tf.int16)
print(num1)

real_num1 = tf.Variable(32.0034, tf.float64)
print(real_num1)

print(tf.rank(num1))

# Tensor reshape
num2 = tf.Variable([12, 32], tf.int16)
print(tf.rank(num2))
print(num2.shape)

mat1 = tf.ones([1, 2, 3])
print(mat1)
print(mat1.shape)

mat2 = tf.reshape(mat1, [2, 3, 1])
print(mat2)

x1 = tf.constant(3, tf.int16)
print(x1)

# Tensor math ops
mat1 = tf.ones([3, 3], tf.int16)
mat2 = tf.ones([3, 3], tf.int16)

mat3 = tf.add(mat1, mat2)
print(mat3)
