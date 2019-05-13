from argparse import ArgumentParser
from os.path import join
from os import linesep
from time import time
from numpy import array, sqrt, average, power, mean
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from td_lambda import td_lambda_estimate
from utilities import pickle_load_or_execute
from experiment_setup import *

from figure4 import get_fig4_data


def draw_figure5(show: bool = False, use_cached_file=False):
    """
    Figure 5: Average error at the best alpha value.
    Uses the same function to generate data for the plot.
    """
    print(linesep + "Generating figure 5 ..." + linesep)
    if not training_sets:
        generate_training_set()
    data = pd.DataFrame(
        pickle_load_or_execute(
            file_path=join('.', 'data', 'fig5.pkl'),
            data_generator=get_fig4_data,
            data_generator_args=([i * 0.05 for i in range(0, 13)], [i * 0.1 for i in range(0, 11)], ground_truth),
            use_cached_file=use_cached_files or use_cached_file),
        columns=["lambda", "alpha", "rms_error"]
    )

    data = data[data.groupby(['lambda'])['rms_error'].transform(min) == data['rms_error']].set_index(keys=['lambda']).drop('alpha', 1)
    plt.figure(num=None, figsize=(7, 6), dpi=120)
    plt.plot(data, marker='o')
    plt.margins(.05)
    plt.xlabel(r"$\lambda$", size=plot_label_size)
    plt.ylabel("ERROR\nUSING\nBEST α", rotation=1, size=plot_label_size, labelpad=25)
    plt.title("Reproduction of Figure 5", size=plot_label_size)
    plt.text(.79, max(data["rms_error"]) * 0.98, "Widrow-Hoff", ha="center", va="center", rotation=0, size=15)

    plt.savefig(join('.', 'figures', 'fig5_{}.png'.format(int(time()))),
                bbox_inches='tight')  # this `bbox_inches` option is necessary to save the plot without clipping the axes labels.
    if show:
        plt.show()

    print(linesep + "Figure 5 done! Find new figures in the `figures` folder!" + linesep)
	
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', help='Specifies if the figures are generated from saved files.', action='store_true')
    parser.add_argument('-s', '--show', help='Displays the figure after plot.', action='store_true')
    args = vars(parser.parse_args())
    draw_figure5(show=args['show'], use_cached_file=args['file'])
