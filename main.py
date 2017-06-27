
# coding: utf-8

# In[1]:

import tensorflow as tf
import prettytensor as pt
import os
import time
import matplotlib.pyplot as mpl
import numpy as np
import matplotlib.image as mpimg
import database


# In[2]:

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot = True)
logs_path = 'tensorboard'
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
save_dir = 'checkpoint/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')


# In[3]:

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

with tf.name_scope('x'):
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name= 'x')
with tf.name_scope('x_image'):
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
with tf.name_scope('y_true'):
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
with tf.name_scope('y_ture_cls'):
    y_true_cls = tf.arg_max(y_true, dimension=1)
with tf.name_scope('x_pretty'):
    x_pretty = pt.wrap(x_image)


# In[4]:

class model():
    def __init__(self):
        with pt.defaults_scope(activation_fn=tf.nn.relu):
            self.y_pred, self.loss = x_pretty.                conv2d(kernel=5, depth=16, name='layer_conv1').                max_pool(kernel=2, stride=2).                conv2d(kernel=5, depth=36, name='layer_conv2').                max_pool(kernel=2, stride=2).                flatten().                fully_connected(size=128, name='layer_fc1').                softmax_classifier(num_classes=num_classes, labels=y_true)

        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
        self.merged_sum = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)
        self.correct_prediction = tf.equal(self.y_pred_cls, y_true_cls)
        with tf.name_scope('acc'):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    
    def optimize(self, num_epoch, load):
        counter = 0
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            for i in range(num_epoch):
                print(i/num_epoch)
                ####print(self.accuracty())
                if load:
                    self.saver.restore(sess = session, save_path = save_path)
                for start, end in zip(range(0, len(database.train_x), 128), range(128, len(database.train_y)+1, 128)):
                    _,rer = session.run([self.optimizer, self.merged_sum], feed_dict = {x: database.train_x[start:end], y_true: database.train_y[start:end]})
                    self.summary_writer.add_summary(rer, (counter))
                    counter +=1
                self.saver.save(sess=session, save_path=save_path)
        print("1.00 ", "done <3")
    def acc(self):
        feed_dict_train = {x: database.test_x, y_true: database.test_y}
        with tf.Session() as session:
            self.saver.restore(sess = session, save_path = save_path)
            acc_train = session.run(self.accuracy, feed_dict=feed_dict_train)
        return acc_train
    def report(self):
        x = self.acc()
        print("\nthe accuracy on the testing dataset is", x ,"good job")
        print("this accuracy was create using a", self.y_pred, "model")



# In[5]:

nn = model()


# In[6]:

nn.optimize(1, load = False)


# In[7]:

nn.optimize(16, load = True)


# In[9]:

nn.report()


# In[ ]:



