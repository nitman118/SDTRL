import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_and_save(path, fname, data):
    """Plot and Save 

    Parameters
    ----------
    path : String
        Relative path of the destination
    name : String
        name of the resulting png file
    data : List
        data to plot
    """
    if not os.path.exists(path):
        os.mkdir(path)
    save_to = os.path.join(path, fname)
    fig, ax = plt.subplots()
    ax.plot(data)
    fig.savefig(save_to)


def save_table(path, fname, table):
    """Save table

    Parameters
    ----------
    table : [Pandas DataFrame]
        [description]
    fname : [String]
        [description]
    """
    res_df = pd.DataFrame(table)
    if not os.path.exists(path):
        os.mkdir(path)
    save_to = os.path.join(path, fname)
    res_df.to_excel(save_to, index=False)
    

def save_state(state, fname):
    np.save(fname, state)

def load_state(fname):
    return np.load(fname)






if __name__ == '__main__':
    plot_and_save('results', 'test.png', [1,2,3,4,3,2,1,3])

    


    



