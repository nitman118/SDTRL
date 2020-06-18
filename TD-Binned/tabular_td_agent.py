import numpy as np
import random

Q_VALUES = {}

class Tabular_TD_agent:

    def __init__(self, discount, step_size, eps, num_actions):
        self.v_vals = {}
        self.discount = discount
        self.step_size = step_size
        self.eps = eps
        self.num_actions = num_actions

    def get_v_value(self, state):
        # print(state)
        # print(type(state))
        state = str(state.tolist())
        if state in self.v_vals:
            return self.v_vals.get(state)
        else:
            self.v_vals[state] = random.random()
            return  self.v_vals[state]
            
    
    def update_v_table(self, state, reward, new_state, done):
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
        #V(state) += stepSize*(R+disc.V(new_state) - V(state))
        state = str(state.tolist()) #stringify state
        if done:
            loss = reward - self.v_vals[state]
            self.v_vals[state] += self.step_size*(loss)
        else:
            # new_state_q_arr = self.get_v_value(new_state)
            new_state_v_val = self.get_v_value(new_state)
            loss = reward + self.discount*new_state_v_val - self.v_vals[state]
            self.v_vals[state] +=  self.step_size*(loss)
        return loss

    

    def set_v_value(self, state):
        # print(state)
        # print(type(state))
        state = str(state.tolist())
        if state not in self.v_vals:
            self.v_vals[state] = random.random()
            





