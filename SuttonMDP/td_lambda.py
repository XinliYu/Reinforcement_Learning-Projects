from random import choice, uniform
import numpy as np


def td_lambda_estimate(alpha, _lambda, state_sequence, weights):
    """
	Calcuates the weight update for the TD(lambda) learning using one provided state sequence (episode).
    :param alpha: the learning rate
    :param _lambda: the lambda parameter for the TD(lambda) estimation
    :param state_sequence: the state sequence (episode) to train the weight update
    :param weights: the current weights
    :return: the weight update derived from the provided `state_sequence`
    """
    delta_weights = np.zeros(7)
    for t in range(0, len(state_sequence) - 1):
        current_state = state_sequence[t]
        next_state = state_sequence[t + 1]
        td = alpha * (weights[next_state] - weights[current_state])
        for k in range(0, t + 1):
            delta_weights[state_sequence[k]] += td * _lambda ** (t - k)

    return delta_weights


def generate_state_sequence(state_count: int = 7, max_len: int = 13, soft_min_len: int = 1):
    """
    Generates one state sequence (episode) for the Sutton paper simulation.
    :param state_count: The number of states.
    :param max_len: The maximum length of an state sequence (episode).
    :param soft_min_len: A soft threshold for the minimum length of the episode. There is still a small possibility for this function to generate one episode of smaller length than this.
    :return: One state sequence (episode).
    """
    state_sequence = [1] * (max_len + 1)
    start_pos = state_count // 2
    if max_len == -1:
        state_sequence = [start_pos]
        while state_sequence[-1] not in [0, state_count - 1]:
            state_sequence.append(state_sequence[-1] + (1 if choice([True, False]) else -1))
    else:
        if uniform(0, 1) > 0.7:
            # if sequence length is not between 4 and 13, then re-sample the sequence
            while len(state_sequence) > max_len or len(state_sequence) < soft_min_len:
                state_sequence = [start_pos]
                while state_sequence[-1] not in [0, state_count - 1]:  # if the current state is not terminal, then randomly go left or right
                    state_sequence.append(state_sequence[-1] + (1 if choice([True, False]) else -1))
        else:
            while len(state_sequence) > max_len:
                state_sequence = [start_pos]
                while state_sequence[-1] not in [0, state_count - 1]:
                    state_sequence.append(state_sequence[-1] + (1 if choice([True, False]) else -1))
    return state_sequence
