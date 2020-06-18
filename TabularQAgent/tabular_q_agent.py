
import numpy as np
import random

Q_VALUES = {}


class Tabular_Q_agent:

    def __init__(self, discount, step_size, eps, num_actions):
        self.q_vals = {}
        self.discount = discount
        self.step_size = step_size
        self.eps = eps
        self.num_actions = num_actions

    def get_q_value(self, state):
        # print(state)
        # print(type(state))
        state = str(state.tolist())
        if state in self.q_vals:
            return self.q_vals.get(state)
        else:
            self.q_vals[state] = [random.random() for _ in range(self.num_actions)]
            return  self.q_vals[state]
            
    
    def update_q_table(self, state, action, reward, new_state, done):
        """[Update the Q Table after observing reward and new_state]

        Parameters
        ----------
        state : [type]
            [description]
        action : [type]
            [description]
        reward : [type]
            [description]
        new_state : [type]
            [description]
        done : function
            [description]
        """
        # self.check_q_exists(state)
        state = str(state.tolist()) #stringify state
        if done:
            loss = reward - self.q_vals[state][action]
            self.q_vals[state][action] += self.step_size*(loss)
        else:
            new_state_q_arr = self.get_q_value(new_state)
            new_state_q_val = np.max(new_state_q_arr)
            loss = reward + self.discount*new_state_q_val - self.q_vals[state][action]
            self.q_vals[state][action] +=  self.step_size*(loss)
        return loss

    def policy(self, state):

        if random.random() < self.eps:
            return random.randint(0,self.num_actions-1)
        else:
            a_vals = self.get_q_value(state)
            action = np.argmax(a_vals)
            return action






