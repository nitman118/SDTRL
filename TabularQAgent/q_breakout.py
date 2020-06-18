import gym
import time
import numpy as np
from utility import plot_and_save, save_table
from tabular_q_agent import Tabular_Q_agent
import datetime

now = datetime.datetime.now()
now_str= now.strftime("%Y-%m-%d-%Hh%Mm%Ss")

NUM_EPISODES = 7500
ENABLE_RENDER = False
EXP_NAME = 'TABULAR_Q_LEARNING'
GYM_ENV = "Breakout-ram-v0"

# env.close()

RES={
    'Length':[],
    'Reward':[],
    'Loss':[],
    'Eps':[]
    }

F_NAME = f'{EXP_NAME}-{GYM_ENV}-{NUM_EPISODES}-{now_str}'


env = gym.make(GYM_ENV)

num_actions = int(env.action_space.n)
num_states = int(env.observation_space.shape[0])

print(env.observation_space.high)
print(env.observation_space.low)

env_space_high = np.array(env.observation_space.high)
env_space_low = np.array(env.observation_space.low)

NUM_INTERVALS = 25



print(f"{GYM_ENV} has {num_actions} actions and {num_states} states")

# Choose an environment from https :// gym . openai . com / envs /# atari .

print(" Press Enter to continue ...")
input()


tabular_q_agent = Tabular_Q_agent(0.99, 0.2, 0.1, num_actions)

episodic_rewards=[]
smooth_ep_rewards=[]
for episode in range(NUM_EPISODES):
    reward_total = 0
    t=0
    loss_total = 0
    ep_length = 0
    obs = env.reset()

    # Run the game .
    done = False
    while not done:
        # env.render ()
        # action = env . action_space.sample()
        action = tabular_q_agent.policy(obs)
        # Take the action , make an observation from the environment and obtain a reward .
        new_obs , reward , done , info = env.step(action)
        loss = tabular_q_agent.update_q_table(obs, action, reward, new_obs, done)
        # dqn_agent.store_experience(obs, action, reward, done, next_obs)
        # print ("At time ",t ,", we obtained reward ", reward)
        obs = new_obs
        t+=1
        reward_total = reward_total + reward
        loss_total += loss
        ep_length+=1

        if ENABLE_RENDER and episode % 3 == 0:
            time.sleep(0.00001)
            env.render()

        if done:
            episodic_rewards.append(reward_total)
            smooth_ep_rewards.append(np.mean(episodic_rewards[-50:]))
            RES['Eps'].append(tabular_q_agent.eps)
            RES['Length'].append(ep_length)
            RES['Reward'].append(reward_total)
            RES['Loss'].append(loss_total)


            print(f"Episode {episode} finished after {t+1} timesteps, running avg reward: {np.mean(episodic_rewards[-10:])}") #, epsilon:{dqn_agent.eps}
            break

env.close()

print(len(tabular_q_agent.q_vals.keys()))

plot_and_save("results","reward_tabular.png",episodic_rewards)
plot_and_save("results","reward_tabular_smooth.png",smooth_ep_rewards)
save_table("results", f"{F_NAME}-data.xlsx", RES)


