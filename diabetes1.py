# In in the case of a fully-connected networks that were simulated, there was significant overfitting
# some getting upto 96.12% accuracy.
# To deal with this kind of problem, I am going to use dropout layer

# DETAILS: Architecture (8-40-30-2)
#          Total Epochs = 30000
#          Data Division - train([:550]) and test([550:])
#          Features used are - all - {'BloodPressure', 'Outcome', 'SkinThickness'}
#          Accuracy without the dropout - 0.747706
#          Accuracy with dropout - 0.796196

# RESULT: Dropout has improved the result

# Step 0: Importing Dependencies
import tensorflow as tf
import pandas as pd

# Step 1: Loading of data
diabetes = pd.read_csv('/path/to/file/diabetes.csv')

# Step 2: Labeling of data + preparing for tensorflow
# This time I am using 'one-hot' system to check weather this works better
diabetes_op = pd.DataFrame()
diabetes_op.loc[:, ('Outcome')] = diabetes.loc[:, ('Outcome')]
diabetes_op.loc[:, ('Outcome_hot_one')] = diabetes_op.loc[:, ('Outcome')] == 0
diabetes_op.loc[:, ('Outcome_hot_one')] = diabetes_op['Outcome_hot_one'].astype(float)
diabetes = diabetes.drop(['Outcome', 'BloodPressure', 'SkinThickness'], axis = 1)
diabetes = diabetes.astype(float)
inputX_train = diabetes[:550].as_matrix()
inputY_train = diabetes_op[:550].as_matrix()

inputX_test = diabetes[400:].as_matrix()
inputY_test = diabetes_op[400:].as_matrix()

# Step 3: Declaring the hyper-parameters + placeholders for input/output
learning_rate = 0.5
training_epochs = 30000
display_step = 1000
keep_prob_1 = tf.placeholder(tf.float32)
keep_prob_2 = tf.placeholder(tf.float32)
n_samples = tf.placeholder(tf.float32)

n_hidden_1 = 40
n_hidden_2 = 30

x = tf.placeholder(tf.float32, shape = [None, 6])
y_ = tf.placeholder(tf.float32, shape = [None, 2])

# Step 4: Making the computation graph

# Layer 1 - Fully Connected - Can use droptouts
W1 = tf.Variable(tf.truncated_normal(shape = [6, n_hidden_1], stddev = 0.1))
b1 = tf.Variable(tf.truncated_normal(shape = [n_hidden_1]))

y1 = tf.add(tf.matmul(x, W1), b1)
y1 = tf.nn.relu(y1)

# Layer 1.5 - dropout
y1_dropout = tf.nn.dropout(y1, keep_prob_1)

# Layer 2 - Fully Connected - Can use dropouts
W2 = tf.Variable(tf.truncated_normal(shape = [n_hidden_1, n_hidden_2], stddev = 0.1))
b2 = tf.Variable(tf.truncated_normal(shape = [n_hidden_2]))

y2 = tf.add(tf.matmul(y1_dropout, W2), b2)
y2 = tf.nn.relu(y2)

# layer 2.5 (per se) - dropouts
y2_dropout = tf.nn.dropout(y2, keep_prob_2)

# Layer 3 - Fully Connected - Cannot use dropouts
W3 = tf.Variable(tf.truncated_normal(shape = [n_hidden_2, 2], stddev = 0.1))
b3 = tf.Variable(tf.truncated_normal(shape = [2]))

y3 = tf.add(tf.matmul(y2_dropout, W3), b3)
y_result = tf.nn.sigmoid(y3)

# Step 5: Training the model
# cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_result))
cross_entropy = tf.reduce_sum(tf.pow(y_ - y_result, 2)) / (2 * n_samples)
# cross_entropy has a much better result than cross_entropy1, simple reason, the data is bianry and so
# using a softmax function will not have a huge impact

optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(training_epochs):
	sess.run([optimizer], feed_dict = {x: inputX_train, y_: inputY_train, keep_prob_1: 0.5, keep_prob_2: 0.5, n_samples: inputY_train.size})

	if i % display_step == 0:
		cross_entropy_value = sess.run([cross_entropy],feed_dict={x: inputX_train, y_: inputY_train, keep_prob_1: 0.5, keep_prob_2: 0.5, n_samples: inputY_train.size})
		print("Step {0} cross_entropy = {1}".format(i, cross_entropy_value[0]))

correct_prediction = tf.equal(tf.arg_max(y_result, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = accuracy.eval(feed_dict = {x: inputX_test, y_: inputY_test, keep_prob_1: 1.0, keep_prob_2: 0.5, n_samples: inputY_test.size}, session = sess)
print("Test Accuracy =",test_accuracy)

sess.close()
