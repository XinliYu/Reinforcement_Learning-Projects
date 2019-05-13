from argparse import ArgumentParser
from numpy import array, zeros, absolute, copy, sqrt, average, power, mean, std
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from os.path import join
from os import linesep
from experiment_setup import *
from td_lambda import td_lambda_estimate
from utilities import pickle_load_or_execute


# Figure 3

def get_fig3_data(alphas, lambdas, truth):
    alphas = array(alphas)
    lambdas = array(lambdas)  # given in figure 3 caption
    state_count = len(truth) + 2

    fig3_data = []

    for lbda in lambdas:
        for alpha in alphas:
            rms_errors = []
            for training_set in training_sets:
                # weights all initialized to zeros except for state "G", according to the paper
                weights = zeros(state_count)
                weights[state_count - 1] = 1.0  # the reward is 1 for state "G"

                while True:
                    prev_weights = copy(weights)
                    delta_weights = zeros(state_count)

                    for sequence in training_set:
                        delta_weights += td_lambda_estimate(alpha, lbda, sequence, weights)

                    weights += delta_weights

                    if sum(absolute(prev_weights - weights)) < 1e-6:  # convergence condition
                        break

                rms_errors.append(sqrt(average(power(truth - weights[1:-1], 2))))

            result = [lbda, alpha, mean(rms_errors), std(rms_errors)]
            fig3_data.append(result)
    return fig3_data


def draw_figure3(show=False, use_cached_file=False):
    print(linesep + "Generating figure 3. This may take half hour, because we are trying different alphas ..." + linesep + 
	"To immediately generate the same figures in the report, run with -f option, i.e. `python3 all_figures.py -f`")
	
    if not training_sets:
        generate_training_set()
    alphas = [0.001, 0.005, 0.01, 0.02, 0.04]
    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    data = pd.DataFrame(
        pickle_load_or_execute(file_path=join('.', 'data', 'fig3.pkl'),
                               data_generator=get_fig3_data,
                               data_generator_args=(alphas, lambdas, ground_truth),
                               use_cached_file=use_cached_file or use_cached_files),
        columns=["lambda", "alphas", "rms_error", "std_error"]
    )

    print(linesep + "---- numerical results for reproducing figure 3 ----" + linesep)
    print(data)
    data.drop('std_error', 1, inplace=True)
    data_by_alphas = data.groupby('alphas')

    curr_data = data_by_alphas.get_group(0.001).set_index(keys=['lambda']).drop("alphas", 1)
    plt.figure(num=None, figsize=(7, 6), dpi=120)
    plt.margins(.05)
    plt.xlabel(r"$\lambda$", size=plot_label_size)
    plt.ylabel("ERROR", size=plot_label_size, labelpad=25).set_rotation(0)
    plt.title("Reproduction of Figure 3", size=plot_label_size)
    plt.xticks([i * .1 for i in range(0, 11)])

    y_tick_scale = 0.005
    y_tick_top = int(max(data["rms_error"]) / y_tick_scale) + 1
    y_tick_bottom = int(min(data["rms_error"]) / y_tick_scale) - 1
    plt.yticks([i * y_tick_scale for i in range(y_tick_bottom, y_tick_top)])

    plt.text(.79, max(curr_data["rms_error"]) * 0.98, "Widrow-Hoff", ha="center", va="center", rotation=0, size=15)
    plt.plot(curr_data, marker='o')
    plt.savefig(join('.', 'figures', 'fig3_{}.png'.format(int(time()))),
                bbox_inches='tight')  # this `bbox_inches` option is necessary to save the plot without clipping the axes labels.

    data.drop(data_by_alphas.get_group(0.001).index, inplace=True)
    data_by_alphas = data.groupby('alphas')

    plot_rows = data_by_alphas.ngroups // 2  # fix up if odd number of groups
    fig, axs = plt.subplots(figsize=(6, 6),
                            nrows=plot_rows, ncols=2,  # fix as above
                            gridspec_kw=dict(hspace=0.3, wspace=0.3))  # Much control of gridspec

    targets = zip(data_by_alphas.groups.keys(), axs.flatten())
    for i, (key, ax) in enumerate(targets):
        curr_data = data_by_alphas.get_group(key).set_index(keys=['lambda']).drop("alphas", 1)
        ax.plot(curr_data, marker='o')
        ax.set_yticks([i * y_tick_scale for i in range(y_tick_bottom + 1, y_tick_top)])
        ax.set_title(r'$\alpha$={}'.format(key))
    plt.savefig(join('.', 'figures', 'fig3_4alphas_subplots_{}.png'.format(int(time()))), bbox_inches='tight')
    if show:
        plt.show()
    plt.clf()

    if show:
        plt.show()
		
    print(linesep + "Figure 3 done! Find new figures in the `figures` folder! " + linesep)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', help='Specifies if the figures are generated from saved files.', action='store_true')
    parser.add_argument('-s', '--show', help='Displays the figure after plot.', action='store_true')
    args = vars(parser.parse_args())
    draw_figure3(show=args['show'], use_cached_file=args['file'])
