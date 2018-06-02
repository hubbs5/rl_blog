# N-Step Advantage Actor Critic Algorithm
# Author: Christian Hubbs
# Email: christiandhubbs@gmail.com

# Simplified n-step actor critic algorithm based on
# Sutton 2018, chapter 13.5

def n_step_a2c(env, actor, critic, n_steps=1, 
               gamma=0.99, num_episodes=2000,
               print_output=True):
    '''
    Inputs
    =====================================================
    env: class, OpenAI environment such as CartPole
    actor: class, parameterized policy network
    critic: class, parameterized value network
    n_steps: integer or string, defines number of steps 
            to take before updating the network. If "MC"
            is passed, it will only update at the end of
            the episode like a Monte Carlo algorithm
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
        
    # Adjust n_steps parameter for 0 indexing
    if type(n_steps) is int:
        n_steps = n_steps - 1
    
    for ep in range(num_episodes):
        
        s_0 = env.reset()
        complete = False
        actions = []
        rewards = []
        states = []
        targets = []
        errors = []
        next_states = []
        current_states = []
        t = 0
        step_counter = 0
        
        while complete == False:
            
            states.append(s_0)
            
            # Select and take action
            action_probs = actor.predict(s_0.reshape(1, -1))
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, complete, _ = env.step(action)
            
            # Calculate predictions and error
            current_state_value = critic.predict(s_0.reshape(1, -1))
            next_state_value = critic.predict(s_1.reshape(1, -1))
            target = r + gamma * next_state_value
            error = target - current_state_value
                
            # Log results
            actions.append(action)
            rewards.append(r)
            targets.append(target)
            errors.append(error)
            next_states.append(next_state_value)
            current_states.append(current_state_value)
            
            # Update parameters once step_counter == n_steps parameter
            if step_counter == n_steps:
                # Convert to arrays and slice to get steps since previous
                # update
                state_array = np.vstack(states)[t - n_steps:t + 1]
                action_array = np.array(actions)[t - n_steps:t + 1]
                reward_array = np.array(rewards)[t - n_steps:t + 1]
                target_array = np.array(targets)[t - n_steps:t + 1]
                error_array = np.array(errors)[t - n_steps:t + 1]
                next_states_array = np.array(next_states)[t - n_steps:t + 1]
                
                # Calculate G_t
                # Discount rewards
                G = sum([gamma**n * reward_array[n] for 
                         n in range(step_counter + 1)])
                # Add discounted future state estimate
                G = G + gamma**(step_counter + 1) * next_state_value
                            
                # Update networks
                actor.update(states=state_array,
                             actions=action_array,
                             returns=error_array)
                critic.update(states=state_array[0].reshape(1, -1),
                              returns=np.array([G]))
            
                step_counter = 0
            else:
                step_counter += 1            
            
            t += 1
            s_0 = s_1
            
            if complete:
                ep_rewards[ep] = np.sum(rewards)
            
            # Print average of last 10 episodes if true
            if print_output and (ep + 1) % 10 == 0 and ep != 1:
                avg_rewards = np.mean(ep_rewards[ep-10:ep+1])
                print("\rA2C with n-steps = {} at Episode: {:d}, Avg Reward: {:.2f}".format(
                        n_steps + 1, ep + 1, avg_rewards), end="")
                
    return ep_rewards