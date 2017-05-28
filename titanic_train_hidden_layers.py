# Loading dependencies
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time 

# To suppress the compiler warning in my computer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Importing function from another python file in directory
from data_loader import load_train_data, get_batch, load_test_data
# Tensorflow version:
print("Tensorflow version is: ", tf.__version__) # written with tf version: 1.1.0
#############################################################

# Defining default hyperparameters
pkeep = 0.75 # probablity of neurons to keep.
learning_rate = 0.0005
training_iterations = 10000 # Training the model n times
batch_size = 100
checkpoint_every = 1000
train_with_only_known_age_data = True

# Number of neurons in hidden layer
# Sizes of each layers (input, 4 hidden and output)
# initial input x_input variables = 5
L = 100
M = 50
N = 25
O = 10
# No need  to use these many layers and large number of neurons.
# I am just trying different possibilities to increase the accuracy by 
# trying all the possible ways, but it seems that the accuracy can't go beyond about 85%
# in average using this model. 
# Probably: Engineering more dependent variables from the given data can help.

# Modeling parameters
tf.flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate (default: 0.0005)')
tf.flags.DEFINE_integer('training_iterations', training_iterations, 'Number of training steps (default: 1000)')
tf.flags.DEFINE_float('pkeep', pkeep, 'Probability of neurons to keep after dropout (default: 0.75)')
tf.flags.DEFINE_integer('batch_size', batch_size, 'size of the input batch (default: 100)')
tf.flags.DEFINE_integer('checkpoint_every', checkpoint_every, 'Save model every this many steps (default: 1000)')
tf.flags.DEFINE_boolean('train_with_only_known_age_data', train_with_only_known_age_data, \
	'training with data with known age (default: True)')
tf.flags.DEFINE_integer("hidden_layer_neurons", {'L': L, 'M': M, 'N': N, 'O': O}, "Number of neurons in the hidden layers")

# printing the default hyperparameters used
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nDefault parameteres used are: ")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value))
print("")

# Training the model n times in batches of size batch_size
def train_model(training_iterations, batch_size, train_data_file):
	accuracy_list, entropy_list = [], []
	x_data, y_data = load_train_data(train_data_file, train_with_only_known_age_data)
	#print("Length of input data is: ", len(x_data))
	start_time = time.time()
	saver = tf.train.Saver()
	for i in range(training_iterations):
		x_batch, y_batch = get_batch(x_data, y_data, batch_size)
		training_data = {x: x_batch, y_: y_batch}
		accrcy, entropy = sess.run([accuracy, cross_entropy], feed_dict=training_data)

		#Backpropagation
		sess.run(train_step, feed_dict=training_data)
		accuracy_list.append(accrcy)
		entropy_list.append(entropy)

		# Saving checkpoints to load trained model later
		directory = "checkpoints/trained_model"
		if not os.path.exists(directory): os.makedirs(directory)
		saver.save(sess, directory, global_step=checkpoint_every)

		# printing the training performance
		if i % 100 == 0:
			print("Accuracy after %s training steps is: %s" % (i, accrcy))
	print("")
	print("Training process is done in time: ", time.time() - start_time, "seconds.")
	return accuracy_list, entropy_list		

# Placeholders
with tf.name_scope("placeholders") as scope:
	x = tf.placeholder(tf.float32, shape=[None, 5])
	y_ = tf.placeholder(tf.float32, shape=[None, 2])


# Defining weight and bias variables
def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	w = tf.Variable(initial, name=name)
	return w

def bias_variable(shape, name):
	initial = tf.ones(shape)/2
	b = tf.Variable(initial, name=name)
	return b

# hidden layer function 
def hidden_layer(x_input, weight, bias, pkeep, activation_function=True):
	# default activation function is: relu in hidden layers
	if activation_function:
		y = tf.matmul(x_input, weight) + bias
		y = tf.nn.relu(y)
		# applying dropout 
		y_out = tf.nn.dropout(y, pkeep)
	else:
		y = tf.matmul(x_input, weight) + bias
		y_out = tf.nn.dropout(y, pkeep)
	return y_out

# Defining the hidden layers in the network
with tf.name_scope("hidden_layer1") as scope:
	W_h1 = weight_variable([5, L], "weight_for_h1")
	b_h1 = bias_variable([L], "bias_for_h1")
	hd_layer1 = hidden_layer(x, W_h1, b_h1, pkeep) 

with tf.name_scope("hidden_layer2") as scope:
	W_h2 = weight_variable([L, M], "weight_for_h2")
	b_h2 = bias_variable([M], "bias_for_h2")
	hd_layer2 = hidden_layer(hd_layer1, W_h2, b_h2, pkeep)

with tf.name_scope("hidden_layer3") as scope:
	W_h3 = weight_variable([M, N], "weight_for_h3")
	b_h3 = bias_variable([N], "bias_for_h3")
	hd_layer3 = hidden_layer(hd_layer2, W_h3, b_h3, pkeep)

with tf.name_scope("hidden_layer4") as scope:
	W_h4 = weight_variable([N, O], "weight_for_h4")
	b_h4 = bias_variable([O], "bias_for_h4")
	hd_layer4 = hidden_layer(hd_layer3, W_h4, b_h4, pkeep)

# The final output layer: softmax layer 
with tf.name_scope("softmax") as scope:
	W = weight_variable([O, 2], "weight_for_softmax")
	b = bias_variable([2], "bias_for_softmax")
	y_logits = tf.matmul(hd_layer4, W) + b
	y = tf.nn.softmax(y_logits)

# accuracy and cross entropy
with tf.name_scope("Accuracy") as scope:
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), axis=1))
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_)
	cross_entropy = tf.reduce_mean(cross_entropy)

# Define a train optimization
with tf.name_scope("train") as scope:
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_step = optimizer.minimize(cross_entropy)

#Session and initialize
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
	
# Data collection 
train_data_file = "train.csv"
accrcy, entropy = train_model(training_iterations, batch_size, train_data_file)

# training progress visualization
training_steps = [i for i in range(training_iterations)]
plt.plot(training_steps, accrcy,  label="Accuracy", c='b')
plt.plot(training_steps, entropy, label="Cross Entropy", c='r')
plt.legend()
plt.xlim(0, training_iterations)
plt.ylim(0, 1.0)
plt.xlabel('Training Steps', color='blue')
plt.ylabel('Accuracy/Cross Entropy', color='blue')
plt.show()

