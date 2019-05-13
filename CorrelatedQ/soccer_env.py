from numpy import array
from enum import Enum
from numpy.random import randint, choice


class SoccerGame:
    class Actions(Enum):
        """
        Enum of player's available actions available during the soccer match
        """
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3
        STICK = 4

    action_count = 5

    def __init__(self, rows=2, cols=4, init_pos_a=(0, 2), init_pos_b=(0, 1), goal_rows=None, init_ball_owner=0):
        self.rows = rows
        self.cols = cols
        self.goal_rows = goal_rows
        self.init_ball_owner = init_ball_owner
        self.init_pos_a = list(init_pos_a) if init_pos_a is not None else None
        self.init_pos_b = list(init_pos_b) if init_pos_b is not None else None
        self.player_pos, self.ball_owner = None, None
        self._cell_count_minus_one = rows * cols - 1
        self._state_count_per_ownership = rows * cols * self._cell_count_minus_one
        self.state_count = self._state_count_per_ownership * 2
        self.available_start_pos = [[i, j] for i in range(rows) for j in range(cols)]
        for goal_row in goal_rows if goal_rows is not None else range(rows):
            self.available_start_pos.remove([goal_row, 0])
            self.available_start_pos.remove([goal_row, cols - 1])
        self.reset()

    def reset(self):
        if self.init_pos_a is None:
            if self.init_pos_b is None:
                pos_idxes = choice(len(self.available_start_pos), 2, replace=False)
                self.player_pos = [self.available_start_pos[pos_idxes[0]], self.available_start_pos[pos_idxes[1]]]
            else:
                self.player_pos = [choice(self.available_start_pos, 1)[0], self.init_pos_b]
        else:
            if self.init_pos_b is None:
                self.player_pos = [self.init_pos_a, choice(self.available_start_pos, 1)[0]]
            else:
                self.player_pos = [self.init_pos_a, self.init_pos_b]
        self.ball_owner = randint(0, 2) if self.init_ball_owner is None else self.init_ball_owner

    def step(self, act_a, act_b, first=None):
        first = randint(0, 2) if first is None else first  # randomly chooses a step that moves player_moves_first
        actions = [act_a, act_b]
        reward = self._move(first, actions[first])
        if reward != 0:
            return reward, first
        return self._move(1 - first, actions[1 - first]), first

    def state(self):
        return self.get_state(self.player_pos[0], self.player_pos[1], self.ball_owner)

    def action_seq(self, length):
        return randint(self.action_count, size=(length, 2)), randint(2, size=length)

    def get_state(self, pos_a, pos_b, ball_owner):
        p1idx = pos_a[0] * self.cols + pos_a[1]
        p2idx = pos_b[0] * self.cols + pos_b[1]
        p2idx -= 1 if p2idx > p1idx else 0
        return ball_owner * self._state_count_per_ownership + p1idx * self._cell_count_minus_one + p2idx

    def _move(self, moving_player, action):
        opponent = 1 - moving_player
        new_pos = self.player_pos[moving_player].copy()
        if action == SoccerGame.Actions.UP.value:
            if new_pos[0] > 0:
                new_pos[0] -= 1
        elif action == SoccerGame.Actions.DOWN.value:
            if new_pos[0] < self.rows - 1:
                new_pos[0] += 1
        elif action == SoccerGame.Actions.LEFT.value:
            if new_pos[1] > 0:
                new_pos[1] -= 1
        elif action == SoccerGame.Actions.RIGHT.value:
            if new_pos[1] < self.cols - 1:
                new_pos[1] += 1

        if new_pos == self.player_pos[opponent]:
            # moves into the opponent's position
            # thenfirst opponent always has the ball
            self.ball_owner = opponent
            return 0
        else:
            self.player_pos[moving_player] = new_pos
            if self.ball_owner == moving_player:
                return self._check_goal(*new_pos)
            else:
                return 0

    def _check_goal(self, row, col):
        if self.goal_rows is None or row in self.goal_rows:
            if col == 0:
                return 100
            elif col == self.cols - 1:
                return -100
        return 0
