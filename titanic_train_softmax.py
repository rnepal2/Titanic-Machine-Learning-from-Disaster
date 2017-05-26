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
learning_rate = 0.0002
beta = 2.0

# Loading train data in the required input form
train_data_file = "train.csv"
variables, survival = load_train_data(train_data_file)

# Training the model n times in batches of size batch_size
def train_model(training_iterations, batch_size):
	accuracy_list, loss_list = [], []
	x_data, y_data = variables, survival
	saver = tf.train.Saver([W, b])
	for i in range(training_iterations):
		x_batch, y_batch = get_batch(x_data, y_data, batch_size)
		training_data = {x: x_batch, y_: y_batch}
		acc, losses = sess.run([accuracy, loss], feed_dict=training_data)
	
		#Backpropagation
		sess.run(train_step, feed_dict=training_data)
		accuracy_list.append(acc)
		loss_list.append(losses)
		
		# Saving some checkpoints to load trained model later
		directory = "checkpoints/trained_model"
		if not os.path.exists(directory): os.makedirs(directory)
		saver.save(sess, directory, global_step=100)
	return accuracy_list, loss_list
				
# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 5])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

#Variables
W = tf.Variable(tf.zeros([5,2]), name='weights')
b = tf.Variable(tf.zeros([2]), name='biases')
y = tf.nn.softmax(tf.matmul(x, W) + b, name='softmax')

# Defining loss and accuracy
is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), axis=1))
# Regularization
regularizer = tf.nn.l2_loss(W)
loss = tf.reduce_mean(loss + beta*regularizer)

# Optimizing the loss by stochastic gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

# Beginning a Session and initializing variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
	
# Running training session
training_iterations = 1000  # Training the model n times
batch_size = 800
acc, losses = train_model(training_iterations, batch_size)

# Visualizing the loss and accuracy of training
training_steps = [i for i in range(training_iterations)]
plt.plot(training_steps, acc,  label="Accuracy", c='b')
plt.plot(training_steps, losses, label="Loss", c='r')
plt.legend()
plt.xlabel("Training Steps")
plt.ylabel("Accuracy/Loss")
plt.show()
