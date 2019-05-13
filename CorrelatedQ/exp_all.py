from mdp_q import *
from ce_q import *
from exp_common import *
from foe_q import *
from friend_q import *

gamma = 0.9
alpha = 0.2
alpha_decay = 0.999995
alpha_min = 0.001
epsilon = 1
epsilon_decay = 0.999995
epsilon_min = 0.001
max_iter = 1000000

exp_common(model_class=CorrQPlayerLP, exp_name='ce-q-lp',
           max_iter=max_iter,
           gamma=gamma,
           alpha=alpha,
           alpha_decay=alpha_decay,
           alpha_min=alpha_min,
           epsilon=epsilon,
           epsilon_decay=epsilon_decay,
           epsilon_min=epsilon_min)

exp_common(model_class=FoeQPlayerLP, exp_name='foe-q-lp',
           max_iter=max_iter,
           gamma=gamma,
           alpha=alpha,
           alpha_decay=alpha_decay,
           alpha_min=alpha_min,
           epsilon=epsilon,
           epsilon_decay=epsilon_decay,
           epsilon_min=epsilon_min)

exp_common(model_class=FriendQPlayer, exp_name='friend-q',
           max_iter=max_iter,
           gamma=gamma,
           alpha=alpha,
           alpha_decay=alpha_decay,
           alpha_min=alpha_min,
           epsilon=epsilon,
           epsilon_decay=epsilon_decay,
           epsilon_min=epsilon_min)

exp_common(model_class=BasicQTwoPlayers, exp_name='basic-q',
           max_iter=max_iter,
           gamma=gamma,
           alpha=alpha,
           alpha_decay=alpha_decay,
           alpha_min=alpha_min,
           epsilon=epsilon,
           epsilon_decay=epsilon_decay,
           epsilon_min=epsilon_min)
