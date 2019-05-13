from exp_common import *
from mdp_q import *

exp_common(model_class=BasicQTwoPlayers, exp_name='basic-q',
           max_iter=1000000,
           test_max_iter=10000,
           progress_msg_interval=5000,
           test_progress_msg_interval=1000,
           gamma=0.9,
           alpha=0.2,
           alpha_decay=0.999995,
           alpha_min=0.001,
           epsilon=1,
           epsilon_decay=0.999995,
           epsilon_min=0.001)
