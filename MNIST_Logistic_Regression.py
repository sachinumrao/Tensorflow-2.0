
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_dir = '/tmp/data/'
num_steps = 100
minibatch_size = 32
learning_rate = 0.2

data = input_data.read_data_sets(data_dir, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784,10]))

y_t = tf.placeholder(tf.float32, [None, 10])

y_p = tf.matmul(x,W)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_p, labels=y_t))

grad_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_p, 1), tf.argmax(y_t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for _ in range(num_steps):
        batch_x , batch_y = data.train.next_batch(minibatch_size)
        sess.run(grad_step, feed_dict={x: batch_x, y_t: batch_y})
        
    # Test the model
    acc = sess.run(accuracy, feed_dict={x:data.test.images, y_t:data.test.labels})

print("Accuracy: ",acc)
