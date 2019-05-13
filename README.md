# RL-Projects

This repository contains three high-quality reinforcement learning course projects.

[**Lunar Lander**](LunarLander): our deep Q-learning model achieves **280+** points on average for the Lunar Lander Problem, the highest score among those we can find online and reported in the class discussion board. Our paper-like report is [here](LunarLander/dqn_lunar_lander.pdf).

>![Best Model Demo](LunarLander/demo.gif)

>![Feedback](LunarLander/feedback.PNG)

[**Correlated-Q**](CorrelatedQ): replicates the results in [Correlated-Q Learning](https://www.aaai.org/Papers/Symposia/Spring/2002/SS-02-02/SS02-02-012.pdf). In addition, we demo the equilibrium evolution. For how to find the linear programming dual, please read our paper-like report [here](CorrelatedQ/reproduction_correlated_q.pdf).

>![CorrelatedQ Replication](CorrelatedQ/imgs/q_diff.PNG) 

[**SuttonMDP**](SuttonMDP): replicates the results in [Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/content/pdf/10.1007/BF00115009.pdf). The same results are not easy to replicate as the paper is vague on the model's parameters. The right parmeter setup is found by repeatedly comparing the charts with the theory. 
