# This file contains two classes for TensorFlow implementations of 
# artificial neural networks used for the running the REINFORCE 
# algorithm as given in chapter 13.3 of Sutton 2017. The first is the 
# policy_estimator network and the second is the value_estimator. 
# Detailed explanation is given at 
# https://www.datahubbs.com/policy-gradients-with-reinforce/

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

class policy_estimator(object):
    
    def __init__(self, sess, env):
        # Pass TensorFlow session object
        self.sess = sess
        # Get number of inputs and outputs from environment
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.learning_rate = 0.01
        
        # Define number of hidden nodes
        self.n_hidden_nodes = 16
        
        # Set graph scope name
        self.scope = "policy_estimator"
        
        # Create network
        with tf.variable_scope(self.scope):
            initializer = tf.contrib.layers.xavier_initializer()
            
            # Define placholder tensors for state, actions, and rewards
            self.state = tf.placeholder(tf.float32, [None, self.n_inputs], 
                                        name='state')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
            self.actions = tf.placeholder(tf.int32, [None], name='actions')
            
            layer_1 = fully_connected(self.state, self.n_hidden_nodes,
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=initializer)
            output_layer = fully_connected(layer_1, self.n_outputs,
                                           activation_fn=None,
                                           weights_initializer=initializer)
            
            # Get probability of each action
            self.action_probs = tf.squeeze(
                tf.nn.softmax(output_layer - tf.reduce_max(output_layer)))
            
            # Get indices of actions
            indices = tf.range(0, tf.shape(output_layer)[0]) \
                * tf.shape(output_layer)[1] + self.actions
                
            selected_action_prob = tf.gather(tf.reshape(self.action_probs, [-1]),
                                             indices)
    
            # Define loss function
            self.loss = -tf.reduce_mean(tf.log(selected_action_prob) * self.rewards)

            # Get gradients and variables
            self.tvars = tf.trainable_variables(self.scope)
            self.gradient_holder = []
            for j, var in enumerate(self.tvars):
                self.gradient_holder.append(tf.placeholder(tf.float32, 
                    name='grads' + str(j)))
            
            self.gradients = tf.gradients(self.loss, self.tvars)
            
            # Minimize training error
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.gradient_holder, self.tvars))
            
    def predict(self, state):
        probs = self.sess.run([self.action_probs], 
                              feed_dict={
                                  self.state: state
                              })[0]
        return probs
    
    def update(self, gradient_buffer):
        feed = dict(zip(self.gradient_holder, gradient_buffer))
        self.sess.run([self.train_op], feed_dict=feed)

    def get_vars(self):
        net_vars = self.sess.run(tf.trainable_variables(self.scope))
        return net_vars

    def get_grads(self, states, actions, rewards):
        grads = self.sess.run([self.gradients], 
            feed_dict={
            self.state: states,
            self.actions: actions,
            self.rewards: rewards
            })[0]
        return grads   


class value_estimator(object):
    
    def __init__(self, sess, env):
        # Pass TensorFlow session object
        self.sess = sess
        # Get number of inputs and outputs from environment
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = 1
        self.learning_rate = 0.01
        
        # Define number of hidden nodes
        self.n_hidden_nodes = 16
        
        # Set graph scope name
        self.scope = "value_estimator"
        
        # Create network
        with tf.variable_scope(self.scope):
            initializer = tf.contrib.layers.xavier_initializer()
            
            # Define placholder tensors for state, actions, and rewards
            self.state = tf.placeholder(tf.float32, [None, self.n_inputs], 
                                        name='state')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
            
            layer_1 = fully_connected(self.state, self.n_hidden_nodes,
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=initializer)
            output_layer = fully_connected(layer_1, self.n_outputs,
                                           activation_fn=None,
                                           weights_initializer=initializer)
            
            self.state_value_estimation = tf.squeeze(output_layer)
    
            # Define loss function as squared difference between estimate and 
            # actual
            self.loss = tf.reduce_mean(tf.squared_difference(
                self.state_value_estimation, self.rewards))

            # Get gradients and variables
            self.tvars = tf.trainable_variables(self.scope)
            self.gradient_holder = []
            for j, var in enumerate(self.tvars):
                self.gradient_holder.append(tf.placeholder(tf.float32, 
                    name='grads' + str(j)))
            
            self.gradients = tf.gradients(self.loss, self.tvars)
            
            # Minimize training error
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.gradient_holder, self.tvars))
            
    def predict(self, state):
        value_est = self.sess.run([self.state_value_estimation], 
                              feed_dict={
                                  self.state: state
                              })[0]
        return value_est
    
    def update(self, gradient_buffer):
        feed = dict(zip(self.gradient_holder, gradient_buffer))
        self.sess.run([self.train_op], feed_dict=feed)

    def get_vars(self):
        net_vars = self.sess.run(tf.trainable_variables(self.scope))
        return net_vars

    def get_grads(self, states, rewards):
        grads = self.sess.run([self.gradients], 
            feed_dict={
            self.state: states,
            self.rewards: rewards
            })[0]
        return grads   