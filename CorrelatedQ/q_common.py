from numpy import zeros, ones, empty, cumsum
from numpy.random import rand, randint, choice
from scipy.optimize import linprog


class QPlayersBase:
    def __init__(self, player_count, state_count, action_count, gamma, alpha, alpha_decay, min_alpha, epsilon, epsilon_decay, min_epsilon):
        self.player_count = player_count
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.state_count = state_count
        self.action_count = action_count
        self.V = self.construct_V()
        self.Q = self.construct_Q()
        self.pi = self.construct_pi()
        self.learning = True
        self._curr_epsilon = epsilon
        self._curr_alpha = alpha

    def construct_V(self):
        pass

    def construct_Q(self):
        pass

    def construct_pi(self):
        pass

    def choose_action(self, state):
        if self.learning and rand() < self._curr_epsilon:
            return randint(self.action_count, size=self.player_count)
        else:
            actions = [None] * self.player_count
            nums = rand(self.player_count)
            for player in range(self.player_count):
                cur_pi = self.pi[player, state]
                p_sum = 0
                for i in range(self.action_count):
                    p = cur_pi[i]
                    if p > 0:
                        p_sum += p
                        if nums[player] < p_sum:
                            actions[player] = i
                            break
                if actions[player] is None:
                    actions[player] = i - 1

            return actions

    def reset(self):
        self._curr_epsilon = self.epsilon
        self._curr_alpha = self.alpha

    def step(self, player, curr_state, next_state, action, reward):
        pass

    def step_all_players(self, curr_state, next_state, action, reward):
        pass

    def update_policy(self, state, V, Q, pi):
        pass

    def get_policy(self, player, state):
        pass


class UncorrelatedQTwoPlayers(QPlayersBase):
    def __init__(self, state_count, action_count, gamma, alpha, alpha_decay, min_alpha, epsilon, epsilon_decay, min_epsilon, rand_init_Q=False, normalized_Q_update=False):
        self._rand_init_Q = rand_init_Q
        super().__init__(player_count=2, state_count=state_count, action_count=action_count, gamma=gamma,
                         alpha=alpha, alpha_decay=alpha_decay, min_alpha=min_alpha,
                         epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        self._curr_alphas = empty(2)
        self._curr_epsilons = empty(2)
        self._curr_alphas.fill(alpha)
        self._curr_epsilons.fill(epsilon)
        self._normalized_Q_update = normalized_Q_update

    def construct_V(self):
        return ones((2, self.state_count))

    def construct_Q(self):
        return rand(2, self.state_count, self.action_count, self.action_count) if self._rand_init_Q else ones((2, self.state_count, self.action_count, self.action_count))

    def construct_pi(self):
        return ones((2, self.state_count, self.action_count)) / self.action_count

    def get_policy(self, player, state):
        return self.pi[player, state]

    def step(self, player, curr_state, next_state, action, reward):
        V, Q, pi = self.V[player], self.Q[player], self.pi[player]

        prev_v = V[next_state]
        self.update_policy(next_state, V, Q, pi)
        post_v = V[next_state]

        alpha = self._curr_alphas[player]
        epsilon = self._curr_epsilons[player]
        Q = Q[curr_state]
        prev_q = Q[action]
        if self._normalized_Q_update:
            Q[action] = (1 - alpha) * prev_q + alpha * ((1 - self.gamma) * reward + self.gamma * post_v)
        else:
            Q[action] = (1 - alpha) * prev_q + alpha * (reward + self.gamma * post_v)
        post_q = Q[action]

        self._curr_alphas[player] = max(self.min_alpha, alpha * self.alpha_decay)
        self._curr_epsilons[player] = max(self.min_epsilon, epsilon * self.epsilon_decay)

        return prev_q, post_q, prev_v, post_v, alpha, epsilon

    def step_all_players(self, curr_state, next_state, action, reward):
        prev_q1, post_q1, prev_v1, post_v1, alpha1, epsilon1 = self.step(0, curr_state, next_state, action, reward[0])
        prev_q2, post_q2, prev_v2, post_v2, alpha2, epsilon2 = self.step(1, curr_state, next_state, (action[1], action[0]), reward[1])
        return (prev_q1, post_q1, prev_v1, post_v1, alpha1, epsilon1), (prev_q2, post_q2, prev_v2, post_v2, alpha2, epsilon2)


class CorrelatedQTwoPlayers(QPlayersBase):
    def __init__(self, state_count, action_count, gamma, alpha, alpha_decay, min_alpha, epsilon, epsilon_decay, min_epsilon, rand_init_Q=False, normalized_Q_update=False):
        self._rand_init_Q = rand_init_Q
        super().__init__(player_count=2, state_count=state_count, action_count=action_count, gamma=gamma,
                         alpha=alpha, alpha_decay=alpha_decay, min_alpha=min_alpha,
                         epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        self._normalized_Q_update = normalized_Q_update

    def construct_V(self):
        return ones((2, self.state_count))

    def construct_Q(self):
        return rand(2, self.state_count, self.action_count, self.action_count) if self._rand_init_Q else ones((2, self.state_count, self.action_count, self.action_count))

    def construct_pi(self):
        return ones((self.state_count, self.action_count, self.action_count)) / (self.action_count ** 2)

    def get_policy(self, player, state):
        return self.pi[state].flatten()

    def step(self, player, curr_state, next_state, action, reward):
        act1, act2 = action
        V1, V2, Q1, Q2 = self.V[0], self.V[1], self.Q[0], self.Q[1]

        prev_v1, prev_v2 = V1[next_state], V2[next_state]
        self.update_policy(next_state, self.V, self.Q, self.pi)
        post_v1, post_v2 = V1[next_state], V2[next_state]

        alpha, epsilon = self._curr_alpha, self._curr_epsilon
        prev_q1, prev_q2 = Q1[curr_state, act1, act2], Q2[curr_state, act1, act2]
        if self._normalized_Q_update:
            Q1[curr_state, act1, act2] = (1 - alpha) * prev_q1 + alpha * ((1 - self.gamma) * reward[0] + self.gamma * post_v1)
            Q2[curr_state, act1, act2] = (1 - alpha) * prev_q2 + alpha * ((1 - self.gamma) * reward[1] + self.gamma * post_v2)
        else:
            Q1[curr_state, act1, act2] = (1 - alpha) * prev_q1 + alpha * (reward[0] + self.gamma * post_v1)
            Q2[curr_state, act1, act2] = (1 - alpha) * prev_q2 + alpha * (reward[1] + self.gamma * post_v2)
        post_q1, post_q2 = Q1[curr_state, act1, act2], Q2[curr_state, act1, act2]

        self._curr_alpha = max(self.min_alpha, alpha * self.alpha_decay)
        self._curr_epsilon = max(self.min_epsilon, epsilon * self.epsilon_decay)

        return prev_q1, post_q1, prev_v1, post_v1, prev_q2, post_q2, prev_v2, post_v2, alpha, epsilon

    def step_all_players(self, curr_state, next_state, action, reward):
        prev_q1, post_q1, prev_v1, post_v1, prev_q2, post_q2, prev_v2, post_v2, alpha, epsilon = self.step(None, curr_state, next_state, action, reward)
        return (prev_q1, post_q1, prev_v1, post_v1, alpha, epsilon), (prev_q2, post_q2, prev_v2, post_v2, alpha, epsilon)
