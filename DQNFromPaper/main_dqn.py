#import gym
import os
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
from utility import save_table, plot_and_save
import datetime


EXP_NAME = 'DQN-Paper'
GYM_ENV = "BreakoutNoFrameskip-v4"
NUM_EPISODES = 5000

now = datetime.datetime.now()
now_str= now.strftime("%Y-%m-%d-%Hh%Mm%Ss")

ENABLE_RENDER = True
# env.close()
RES={
    'Length':[],
    'Reward':[],
    'Loss':[],
    'Eps':[]
    }

F_NAME = f'{EXP_NAME}-{GYM_ENV}-{NUM_EPISODES}-{now_str}'


if __name__ == '__main__':
    env = make_env('BreakoutNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    n_games = NUM_EPISODES
    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=75000, eps_min=0.1,
                     batch_size=32, replace=1000, eps_dec=0.2e-4,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='BreakoutNoFrameskip-v4')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'

    
    
    figure_file = os.path.join('plots',rf'{fname}.png')

    print(fname)
    print(figure_file)

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):

        if (i+1) % 100 == 0:
            print('----Saving Intermediate Result----')
            F_NAME = f'{EXP_NAME}-{GYM_ENV}-{i}-{now_str}'
            save_table('results',f'{F_NAME}.xlsx', RES)
        

        done = False
        observation = env.reset()
        reward_total = 0
        loss_total = 0
        ep_length = 0

        score = 0
        while not done:
            if i%10==0:
                env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, int(done))
                loss = agent.learn()
            observation = observation_
            reward_total += reward
            ep_length += 1
            loss_total += loss

        RES['Eps'].append(agent.epsilon)
        RES['Reward'].append(reward_total)
        RES['Length'].append(ep_length)
        RES['Loss'].append(loss_total)

        avg_score = np.mean(RES['Reward'][-100:])
        print('episode: ', i,'score: ', reward_total,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', ep_length)

        if avg_score > best_score:
            if not load_checkpoint:
               agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if load_checkpoint and n_steps >= 18000:
            break

    x = [i+1 for i in range(len(scores))]
    # plot_learning_curve(steps_array, scores, eps_history, figure_file)
    
    F_NAME = f'{EXP_NAME}-{GYM_ENV}-{NUM_EPISODES}-{now_str}'
    save_table('results',f'{F_NAME}.xlsx', RES)
    # plot_and_save('results', fname= , data=)