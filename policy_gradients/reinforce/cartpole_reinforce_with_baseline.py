# Test the REINFORCE algorithm with baseline using OpenAI Gym"s cartpole 
# environment.
# For details see: 
# https://www.datahubbs.com/policy-gradients-with-reinforce/

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import gym
from time import time

# Import custom functions from this repo
from networks import policy_estimator, value_estimator
from reinforce_with_baseline import *

env = gym.make("CartPole-v0")
tf.reset_default_graph()
sess = tf.Session()

pe = policy_estimator(sess, env)
ve = value_estimator(sess, env)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

print("\nBegin Training...")

start_time = time()
rewards = reinforce_baseline(env, pe, ve)
end_time = time()

print("Training Time: {:.2f}".format(end_time - start_time))

smoothed_rewards = [np.mean(rewards[max(0,i-10):i+1]) for i in
 range(len(rewards))]

plt.figure()
plt.plot(smoothed_rewards)
plt.title("CartPole Rewards using REINFORCE with Policy Baseline")
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.show()