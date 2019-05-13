from numpy import argmax, unravel_index
from q_common import *


class BasicQTwoPlayers(UncorrelatedQTwoPlayers):
    def construct_Q(self):
        return ones((2, self.state_count, self.action_count))

    def update_policy(self, state, V, Q, pi):
        Q1 = Q[state]
        idx = argmax(Q1)
        V[state] = Q1[idx]
        pi[state] = 0
        pi[state, idx] = 1

    def step_all_players(self, curr_state, next_state, action, reward):
        prev_q1, post_q1, prev_v1, post_v1, alpha1, epsilon1 = self.step(0, curr_state, next_state, action[0], reward[0])
        prev_q2, post_q2, prev_v2, post_v2, alpha2, epsilon2 = self.step(1, curr_state, next_state, action[1], reward[1])
        return (prev_q1, post_q1, prev_v1, post_v1, alpha1, epsilon1), (prev_q2, post_q2, prev_v2, post_v2, alpha2, epsilon2)
