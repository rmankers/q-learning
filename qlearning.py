import gym
import numpy as np
import sys

env = gym.make('Acrobot-v1',new_step_api=False,max_episode_steps = 500)

LEARNING_RATE = 0.1
DISCOUNT = 0.95
DISCRETE_OS_SIZE = [10] * len(env.observation_space.high)
EPISODES = 25000
SHOW_EVERY = 5000

discrete_os_win_size = (env.observation_space.high + 0.02 - env.observation_space.low) / DISCRETE_OS_SIZE
q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):

    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple((np.floor(discrete_state)).astype(int))
  
for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    done = False
    i = 0
    current_state = env.reset()

    while not done:

        current_discrete_state = get_discrete_state(current_state)

        # Choosing next action based on maximum q of current discrete state
        action = q_table[current_discrete_state].argmax()  

        # Telling agent to take action, and recording new state/reward/done
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        


        # Update Q
        current_q = q_table[current_discrete_state + (action,)]
        max_future_q = q_table[new_discrete_state].max()
        updated_q = (1-LEARNING_RATE) * current_q  + LEARNING_RATE * (reward + DISCOUNT *max_future_q)

        q_table[current_discrete_state + (action,)] = updated_q

        current_state = new_state
                
        if render:        
            env.render()  
  
