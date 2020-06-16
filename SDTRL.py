import gym
import time
from agent import Agent
import numpy as np
from utility import plot_and_save

NUM_EPISODES = 100
# env.close()


GYM_ENV = "Assault-ram-v0"
#GYM_ENV="Pong-ram-v0"
#GYM_ENV="LunarLander-v2"
GYM_ENV = "MountainCar-v0"
GYM_ENV = "CartPole-v0"
# GYM_ENV = "Breakout-ram-v0"
env = gym.make(GYM_ENV)


num_actions = int(env.action_space.n)
num_states = int(env.observation_space.shape[0])

print(f"{GYM_ENV} has {num_actions} actions and {num_states} states")

dqn_agent = Agent(num_states=num_states, num_actions=num_actions, eps=1, eps_decay = 0.001, batch_size = 32, discount=0.99, learning_rate=0.025, replay_buffer_cap=1000000)

# Choose an environment from https :// gym . openai . com / envs /# atari .

print(" Press Enter to continue ...")
input()

episodic_rewards = []
smooth_ep_rewards = []
episodic_loss = []
smooth_ep_loss = []


for episode in range(NUM_EPISODES):
    reward_total = 0
    loss_total = 0
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

        if episode > 3:
            loss = dqn_agent.update_q_function()
            loss_total+=loss

        if episode % 10 == 0:
            env.render()

        if done:
            episodic_rewards.append(reward_total)
            episodic_loss.append(loss_total)
            smooth_ep_rewards.append(np.mean(episodic_rewards[-10:]))
            smooth_ep_loss.append(np.mean(episodic_loss[-10:]))
            print(f"Episode {episode} finished after {t+1} timesteps, running avg reward: {np.mean(episodic_rewards[-10:])}, epsilon:{dqn_agent.eps}")
            break

env.close()
plot_and_save("results","dqn_reward_tabular.png",episodic_rewards)
plot_and_save("results","dqn_tabular_smooth.png",smooth_ep_rewards)
plot_and_save("results","dqn_loss.png",episodic_loss)
plot_and_save("results","dqn_loss_smooth.png",smooth_ep_loss)
