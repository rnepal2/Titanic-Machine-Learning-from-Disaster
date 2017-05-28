# Loading dependencies
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# To suppress the compiler warning in my computer
# You may not require this in your computer.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Importing function from another python file in directory
from data_loader import load_train_data, get_batch
# Tensorflow version:
print("Tensorflow version is: ", tf.__version__)
#############################################################

# Hyperparameters
learning_rate = 0.0005
training_iterations = 10000  # Training the model n times
batch_size = 500
beta = 0.10

# Loading train data in the required input form
# Training the model n times in batches of size batch_size
def train_model(training_iterations, batch_size, train_data_file):
	accuracy_list, cross_entropy_list = [], []
	xs_data, ys_data = load_train_data(train_data_file, True)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for i in range(training_iterations):
			x_batch, y_batch = get_batch(xs_data, ys_data, batch_size)
			training_data = {x: x_batch, y_: y_batch}
			accrcy, s_cross = sess.run([accuracy, cross_entropy], feed_dict=training_data)

			#Backpropagation
			sess.run(train_step, feed_dict=training_data)
			accuracy_list.append(accrcy)
			cross_entropy_list.append(s_cross)
		return accuracy_list, cross_entropy_list
				
# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 5])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# Variables
W = tf.Variable(tf.zeros([5,2]), name='weights')
b = tf.Variable(tf.zeros([2]), name='biases')
y_logits = tf.matmul(x, W) + b
y = tf.nn.softmax(y_logits, name='softmax')

# Defining cross entropy/loss and accuracy
is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), axis=1))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_)

# Regularization
regularizer = tf.nn.l2_loss(W)
cross_entropy = tf.reduce_mean(cross_entropy) + beta*regularizer

# Optimization
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

# Data oollection and training progress visualization
train_data_file = "train.csv"
accrcy, s_cross = train_model(training_iterations, batch_size, train_data_file)

training_steps = [i for i in range(training_iterations)]
plt.plot(training_steps, accrcy,  label="Accuracy", c='b')
plt.plot(training_steps, s_cross, label="Cross Entropy", c='r')
plt.legend()
plt.xlabel("Training Steps", color='magenta')
plt.ylabel("Accuracy/Cross Entropy", color='magenta')
plt.xlim(0, training_iterations)
plt.ylim(0, 1)
plt.show()
