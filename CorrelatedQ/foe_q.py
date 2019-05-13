from numpy import array, zeros, ones, empty, vstack, eye
from scipy.optimize import linprog
from q_common import *
from cvxopt import matrix, solvers
from cvxopt.modeling import op, variable


class FoeQPlayerLP(UncorrelatedQTwoPlayers):
    def update_policy(self, state, V, Q, pi):
        act_count = self.action_count
        c = zeros(act_count + 1)
        c[0] = -1.0
        G = empty((act_count, act_count + 1))
        G[:, 1:] = -Q[state].T
        G[:, 0] = 1
        G = vstack((-eye(act_count + 1, act_count + 1), G))
        G[0, 0] = 0

        h = zeros(act_count + act_count + 1)
        A = ones((1, act_count + 1))
        A[0, 0] = 0
        b = [1.]

        res = solvers.lp(matrix(c), matrix(G), matrix(h), matrix(A), matrix(b))
        x = array(res['x']).flatten()
        pi[state] = x[1:]
        V[state] = x[0]


class FoeQPlayerModelOP(UncorrelatedQTwoPlayers):
    def update_policy(self, state, V, Q, pi):
        op, v, var_pi = self._create_op(state, Q)
        op.solve()
        V[state] = v.value[0]
        for i in range(self.action_count):
            pi[state, i] = var_pi[i].value[0]

    def _create_op(self, state, Q):
        var_pi = [variable() for _ in range(self.action_count)]
        constraints = [(var_pi[i] >= 0) for i in range(self.action_count)]
        constraints.append((sum(var_pi) == 1))
        v = variable()
        for j in range(self.action_count):
            c = 0
            for i in range(self.action_count):
                c += float(Q[state, i, j]) * var_pi[i]
            constraints.append((c >= v))
        return op(-v, constraints), v, var_pi
