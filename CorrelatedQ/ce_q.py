from numpy import array, zeros, ones, empty, vstack, eye
from numpy import sum as npsum
from scipy.optimize import linprog
from q_common import *
from cvxopt import matrix, solvers
from cvxopt.modeling import op, variable


class CorrQPlayerLP(CorrelatedQTwoPlayers):
    def update_policy(self, state, V, Q, pi):
        act_len = self.action_count
        x_len = act_len ** 2
        Q1 = Q[0, state]
        Q2 = Q[1, state]

        c = (Q1 + Q2).flatten()

        G = zeros(((act_len - 1) * act_len * 2, x_len))
        G_row_idx = 0
        G_col_idx = 0
        next_G_col_idx = act_len
        for i_pivot in range(act_len):
            for i in range(act_len):
                if i != i_pivot:
                    G[G_row_idx, G_col_idx:next_G_col_idx] = Q1[i] - Q1[i_pivot]
                    G_row_idx += 1
            G_col_idx = next_G_col_idx
            next_G_col_idx += act_len

        G_col_idx = 0
        for j_pivot in range(act_len):
            for j in range(act_len):
                if j != j_pivot:
                    G[G_row_idx, range(G_col_idx, x_len, act_len)] = Q2[:, j] - Q2[:, j_pivot]
                    G_row_idx += 1
            G_col_idx += 1
        G = vstack((-eye(x_len, x_len), G))

        h = zeros(G.shape[0])
        A = ones((1, x_len))
        b = [1.]

        res = solvers.lp(matrix(c), matrix(G), matrix(h), matrix(A), matrix(b))
        x = array(res['x']).flatten()
        pi[state] = x.reshape((act_len, act_len))

        V[0, state] = npsum(pi[state] * Q1)
        V[1, state] = npsum(pi[state] * Q2)


class CorrQPlayerModelOP(CorrelatedQTwoPlayers):
    def update_policy(self, state, V, Q, pi):
        V1, V2, Q1, Q2 = V[0], V[1], Q[0], Q[1]
        op, v, var_pi = self._create_op(state, Q1, Q2)
        op.solve()

        if op.status == 'optimal':
            v1 = 0
            for i in range(self.action_count):
                for j in range(self.action_count):
                    pi[state, i, j] = var_pi[i][j].value[0]
                    v1 += var_pi[i][j].value[0] * float(Q1[state, i, j])
            V1[state] = v1

            v2 = 0
            for i in range(self.action_count):
                for j in range(self.action_count):
                    v2 += var_pi[i][j].value[0] * float(Q2[state, i, j])
            V2[state] = v2

    def _create_op(self, state, Q1, Q2):
        var_pi = [[variable() for _ in range(self.action_count)] for __ in range(self.action_count)]
        sum_var_pi = 0
        constraints = []
        for i in range(self.action_count):
            for j in range(self.action_count):
                constraints.append(var_pi[i][j] >= 0)
                sum_var_pi += var_pi[i][j]
        constraints.append((sum_var_pi == 1))
        v = variable()

        for i in range(self.action_count):
            rc1 = 0
            for j in range(self.action_count):
                rc1 += var_pi[i][j] * float(Q1[state, i, j])

            for k in range(self.action_count):
                if i != k:
                    rc2 = 0
                    for j in range(self.action_count):
                        rc2 += var_pi[i][j] * float(Q1[state, k, j])
                    constraints.append((rc1 >= rc2))

        for i in range(self.action_count):
            rc1 = 0
            for j in range(self.action_count):
                rc1 += var_pi[j][i] * float(Q2[state, j, i])

            for k in range(self.action_count):
                if i != k:
                    rc2 = 0
                    for j in range(self.action_count):
                        rc2 += var_pi[j][i] * float(Q2[state, j, k])
                    constraints.append((rc1 >= rc2))

        sum_total = 0
        for i in range(self.action_count):
            for j in range(self.action_count):
                sum_total += var_pi[i][j] * float(Q1[state, i, j])
                sum_total += var_pi[i][j] * float(Q2[state, i, j])

        constraints.append((v == sum_total))

        return op(-v, constraints), v, var_pi
