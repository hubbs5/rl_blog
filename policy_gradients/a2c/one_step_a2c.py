# One-Step Advantage Actor Critic Algorithm
# Author: Christian Hubbs
# Email: christiandhubbs@gmail.com

# Simplified one-step actor critic algorithm from
# Sutton 2018, chapter 13.5

import numpy as np

def one_step_a2c(env, actor, critic, gamma=0.99, 
                 num_episodes=2000, print_output=True):
    '''
    Inputs
    =====================================================
    env: class, OpenAI environment such as CartPole
    actor: class, parameterized policy network
    critic: class, parameterized value network
    gamma: float, 0 < gamma <= 1, determines the discount
            factor to be applied to future rewards.
    num_episodes: int, the number of episodes to be run.
    print_output: bool, prints algorithm settings and 
            average of last 10 episodes to track training.
    
    Outputs
    ======================================================
    ep_rewards: np.array, sum of rewards for each 
            simulated episode
    '''
    
    # Set up vectors for episode data
    ep_rewards = np.zeros(num_episodes)
    
    action_space = np.arange(env.action_space.n)
    
    for ep in range(num_episodes):
        
        s_0 = env.reset()
        complete = False
        actions = []
        rewards = []
        states = []
        targets = []
        errors = []
        t = 0
        
        while complete == False:
            
            # Select and take action
            action_probs = actor.predict(s_0.reshape(1, -1))
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, complete, _ = env.step(action)
            
            # Calculate predictions and error
            predicted_value = critic.predict(s_1.reshape(1, -1))
            target = r + gamma * predicted_value
            error = target - critic.predict(s_0.reshape(1, -1))
                
            # Log results
            states.append(s_0)
            actions.append(action)
            rewards.append(r)
            targets.append(target)
            errors.append(error)
            
            # Update networks
            actor.update(states=np.array(states),
                         actions=np.array(actions),
                         returns=np.array(errors))
            critic.update(states=np.array(states),
                          returns=np.array(targets))
            
            t += 1
            s_0 = s_1
            
            if complete:
                ep_rewards[ep] = np.sum(rewards)
                
                # Print average of last 10 episodes if true
                if print_output and (ep + 1) % 10 == 0 and ep != 1:
                    avg_rewards = np.mean(ep_rewards[ep-10:ep+1])
                    print("\rOne-step A2C at Episode: {:d}, Avg Reward: {:.2f}".format(
                            ep + 1, avg_rewards), end="")
                
    return ep_rewards