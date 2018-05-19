# This file contains the REINFORCE with baseline algorithm.
# For details see: 
# https://www.datahubbs.com/policy-gradients-with-reinforce/

import numpy as np

def reinforce_baseline(env, policy_estimator, value_estimator,
                       num_episodes=2000, batch_size=10, gamma=0.99):
    
    total_rewards = []
    
    # Set up gradient buffers and set values to 0
    # Policy estimation buffer
    grad_buffer_pe = policy_estimator.get_vars()
    for i, g in enumerate(grad_buffer_pe):
        grad_buffer_pe[i] = g * 0
    # Value estimation buffer
    grad_buffer_ve = value_estimator.get_vars()
    for i, g in enumerate(grad_buffer_ve):
        grad_buffer_ve[i] = g * 0
        
    # Get possible actions
    action_space = np.arange(env.action_space.n)
        
    for ep in range(num_episodes):
        # Get initial state
        s_0 = env.reset()
        reward = 0
        episode_log = []
        # Log value estimation
        complete = False
        
        # Run through each episode
        while complete == False:
            
            # Get the probabilities over the actions
            action_probs = policy_estimator.predict(
                s_0.reshape(1,-1))
            
            # Estimate the value
            value_est = value_estimator.predict(
                s_0.reshape(1,-1))
            
            # Stochastically select the action
            action = np.random.choice(action_space,
                                      p=action_probs)
            # Take a step
            s_1, r, complete, _ = env.step(action)
            
            # Calculate reward-estimation delta
            re_delta = r - value_est
            
            # Append results to the episode log
            episode_log.append([s_0, action, re_delta, r, s_1])
            s_0 = s_1
            
            # If complete, store results and calculate the gradients
            if complete:
                episode_log = np.array(episode_log)
                
                # Store raw rewards and discount reward-estimation delta
                total_rewards.append(episode_log[:,3].sum())
                discounted_rewards = discount_rewards(
                    episode_log[:,3], gamma)
                discounted_reward_est = discount_rewards(
                    episode_log[:,2], gamma)
                
                # Calculate the gradients for the policy estimator and
                # add to buffer
                pe_grads = policy_estimator.get_grads(
                    states=np.vstack(episode_log[:,0]),
                    actions=episode_log[:,1],
                    rewards=discounted_rewards)
                for i, g in enumerate(pe_grads):
                    grad_buffer_pe[i] += g
                    
                # Calculate the gradients for the value estimator and
                # add to buffer
                ve_grads = value_estimator.get_grads(
                    states=np.vstack(episode_log[:,0]),
                    rewards=discounted_reward_est)
                for i, g in enumerate(ve_grads):
                    grad_buffer_ve[i] += g
                    
        # Update policy gradients based on batch_size parameter
        if ep % batch_size == 0 and ep != 0:
            policy_estimator.update(grad_buffer_pe)
            value_estimator.update(grad_buffer_ve)
            
            # Clear buffer values for next batch
            for i, g in enumerate(grad_buffer_pe):
                grad_buffer_pe[i] = g * 0
                
            for i, g in enumerate(grad_buffer_ve):
                grad_buffer_ve[i] = g * 0
                
    return total_rewards

# Function for discounting rewards received by the agent
def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards