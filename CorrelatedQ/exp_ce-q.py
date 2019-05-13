from ce_q import *
from exp_common import *

use_cvxopt_model = False
if use_cvxopt_model:
    model_class = CorrQPlayerModelOP
    exp_name = 'ce-q-op'
else:
    model_class = CorrQPlayerLP
    exp_name = 'ce-q-lp'

for rand_init_Q in (True, False):
    for normalized_Q_update in (True, False):
        for gamma, alpha in ((0.9, 0.2),):
            exp_common(model_class=model_class, exp_name=exp_name,
                       test_mode=False,
                       max_iter=1000000,
                       test_max_iter=10000,
                       progress_msg_interval=5000,
                       test_progress_msg_interval=1000,
                       gamma=gamma,
                       alpha=alpha,
                       alpha_decay=0.999995,
                       alpha_min=0.001,
                       epsilon=1,
                       epsilon_decay=0.999995,
                       epsilon_min=0.001,
                       rand_init_Q=rand_init_Q,
                       normalized_Q_update=normalized_Q_update)
