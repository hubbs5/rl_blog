# Test the REINFORCE algorithm with OpenAI Gym's cartpole environment.
# For details see: 
# https://www.datahubbs.com/policy-gradients-with-reinforce/

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import gym
from time import time

# Import custom functions from this repo
from networks import policy_estimator
from reinforce import *

env = gym.make('CartPole-v0')
tf.reset_default_graph()
sess = tf.Session()

pe = policy_estimator(sess, env)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

print("\nBegin Training...")

start_time = time()
rewards = reinforce(env, pe)
end_time = time()
print("Training Time: {:.2f}".format(end_time - start_time))

# Smooth rewards
smoothed_rewards = [np.mean(rewards[max(0,i-10):i+1]) for i in
 range(len(rewards))]

# Plot results
plt.figure()
plt.plot(smoothed_rewards)
plt.title("CartPole Rewards with REINFORCE")
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.show()