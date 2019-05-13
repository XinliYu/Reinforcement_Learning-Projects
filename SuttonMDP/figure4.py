from os.path import join
from os import linesep
from time import time
from argparse import ArgumentParser
from numpy import array, sqrt, average, power, mean
import pandas as pd
import matplotlib.pyplot as plt
from td_lambda import td_lambda_estimate
from utilities import pickle_load_or_execute
from experiment_setup import *


def get_fig4_data(alphas, lambdas, truth):
    alphas = array(alphas)
    lambdas = array(lambdas)
    non_terminal_state_count = len(truth)

    fig4_data = []

    for lbda in lambdas:
        for alpha in alphas:
            rms_errors = []
            for training_set in training_sets:
                # initialization of weights;
                # the first state has weight 0, the last state has weight 1, and all others are 0.5
                weights = array([0.] + [0.5 for _ in range(non_terminal_state_count)] + [1.])

                # update the weights by `td_lambda_estimate`
                for sequence in training_set:
                    delta_weights = td_lambda_estimate(alpha, lbda, sequence, weights)
                    weights += delta_weights

                rms_errors.append(sqrt(average(power(ground_truth - weights[1:-1], 2))))

            result = [lbda, alpha, mean(rms_errors)]
            fig4_data.append(result)
    return fig4_data


def draw_figure4(repeat: int = 1, show: bool = False, use_cached_file=False):
    print(linesep + "Generating figure 4 ..." + linesep)
    for iter_idx in range(repeat):
        data = pd.DataFrame(
            pickle_load_or_execute(
                file_path=join('.', 'data', 'fig4.pkl'),
                data_generator=get_fig4_data,
                data_generator_args=([i * 0.05 for i in range(0, 13)], [0.0, 0.3, 0.8, 1], ground_truth),
                use_cached_file=(use_cached_files or use_cached_file) and repeat == 1),
            columns=["lambda", "alpha", "rms_error"]
        )

        print(linesep + "---- numerical results for reproducing figure 4 & 5 ----" + linesep)
        print(data)

        fig, ax = plt.subplots(figsize=(8, 6))
        # creates a alpha-lambda matrix with rows indexed by `alpha` and columns indexed by `lambda`
        # then it plots for every column of that this
        data.groupby(['alpha', 'lambda']) \
            .sum()['rms_error'] \
            .unstack() \
            .plot(ax=ax, marker='o')
        ax.set_xlabel(r'$\alpha$', size=plot_label_size)
        ax.set_ylabel('ERROR', size=plot_label_size, labelpad=25).set_rotation(0)
        plt.title("Reproduction of Figure 4", size=plot_label_size)
        plt.savefig(join('.', 'figures', 'fig4_{}.png'.format(int(time()))),
                    bbox_inches='tight')  # this `bbox_inches` option is necessary to save the plot without clipping the axes labels.
        if show:
            plt.show()
        if iter_idx != repeat - 1:
            generate_training_set()
    print(linesep + "Figure 4 done! Find new figures in the `figures` folder! " + linesep)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--repeat', help='Specifies how many figures are generated in one run.', default=1)
    parser.add_argument('-f', '--file', help='Specifies if the figures are generated from saved files.', action='store_true')
    parser.add_argument('-s', '--show', help='Displays the figure after plot.', action='store_true')
    args = vars(parser.parse_args())
    draw_figure4(args['repeat'], show=args['show'], use_cached_file=args['file'])
