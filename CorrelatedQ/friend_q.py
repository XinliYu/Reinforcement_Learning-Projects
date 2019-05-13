from numpy import argmax, unravel_index
from q_common import *


class FriendQPlayer(UncorrelatedQTwoPlayers):
    def update_policy(self, state, V, Q, pi):
        Q1 = Q[state]
        idx = unravel_index(argmax(Q1, axis=None), Q1.shape)
        V[state] = Q1[idx]
        pi[state] = 0
        pi[state, idx[0]] = 1
