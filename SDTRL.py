import gym
import time
from agent import Agent
import numpy as np

NUM_EPISODES = 100
# env.close()

GYM_ENV = "Assault-ram-v0"
GYM_ENV="Pong-ram-v0"
#GYM_ENV="LunarLander-v2"
GYM_ENV = "CartPole-v0"
env = gym.make(GYM_ENV)


num_actions = int(env.action_space.n)
num_states = int(env.observation_space.shape[0])

print(f"{GYM_ENV} has {num_actions} actions and {num_states} states")

dqn_agent = Agent(num_states=num_states, num_actions=num_actions, eps=1, eps_decay = 0.01,batch_size=32, discount=0.9995, learning_rate=0.003, replay_buffer_cap=1000000)

# Choose an environment from https :// gym . openai . com / envs /# atari .

print(" Press Enter to continue ...")
input()

rewards = []


for episode in range(NUM_EPISODES):
    reward_total = 0
    t=0
    obs = env.reset()

    # Run the game .
    done = False
    while not done:
        # env.render ()
        # action = env . action_space.sample()
        action = dqn_agent.policy(obs)
        # Take the action , make an observation from the environment and obtain a reward .
        next_obs , reward , done , info = env.step(action)
        dqn_agent.store_experience(obs, action, reward, done, next_obs)
        # print ("At time ",t ,", we obtained reward ", reward)
        obs = next_obs
        t+=1
        reward_total = reward_total + reward

        if episode > 2:
            dqn_agent.update_q_function()

        if episode % 10 == 0:
            env.render()

        if done:
            rewards.append(reward_total)
            print(f"Episode finished after {t+1} timesteps, running avg reward: {np.mean(rewards[-10:])}, epsilon:{dqn_agent.eps}")
            break

env.close()
