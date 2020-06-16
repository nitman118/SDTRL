import os
import matplotlib.pyplot as plt


def plot_and_save(path, name, data):
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
    save_to = os.path.join(path, name)
    fig, ax = plt.subplots()
    ax.plot(data)
    fig.savefig(save_to)


if __name__ == '__main__':
    plot_and_save('results', 'test.png', [1,2,3,4,3,2,1,3])

    


    



