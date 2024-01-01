# Reinforcement Learning on Gym MountainCar Experiment

## Introduction:

Reinforcement learning (RL) is a branch of machine learning that focuses on training agents to make sequential decisions in an environment to maximize a cumulative reward. In this experiment, I applied RL Q-Table techniques to solve the MountainCar problem using the OpenAI Gym framework.

## Experimental Setup:

### How to Run :

```
pip3 install -r requirements.txt
python3 main.py
```

### Environment:

I utilized the Gym MountainCar environment, which presents a simple physics-based problem. The agent, represented as a car, must learn to drive up a steep hill by building momentum in a valley.

### Algorithm:

My experimentation involved implementing a Q-learning algorithm to train the agent. The Q-learning algorithm is a model-free RL technique that learns the optimal action-value function for making decisions.

## Methodology:

### Q-Learning Implementation:

I implemented Q-learning with a Q-table to represent state-action pairs. The key components of our implementation included state representation, action selection, reward calculation, and Q-value updates.
In this environment state has two values. Position and velocity. This two values are continious and we need to convert it to discrete value. I used state aggregation method to make states discrete for adapting and representing it to Q-Table.

## Parameter Exploration:

### Learning Rate (α):

I systematically varied the learning rate (α) to observe its impact on the convergence and learning speed of the Q-learning algorithm. Different values were tested, and the results were analyzed to understand the trade-offs. There are graphics with different Learning Rate Value Converging on conclusion section.

### Discount Factor (γ):

Various values of the discount factor (γ) were experimented with to investigate how they influenced the agent's balance between short-term and long-term rewards.There are graphics with different Discount Factor Value Converging on conclusion section..

### Exploration vs. Exploitation (ε-greedy):

I explored different exploration rates using an ε-greedy strategy to understand the trade-off between exploration and exploitation. This parameter's impact on learning efficiency and convergence was carefully analyzed.

## Challenges:

It was hard to convert continous state space to discrete state space. Visualizing the results and parameter affects was also challenging. I needed help to speed up to environment by disabling visualization. Otherwise train process advance too slowly.


## Conclusion:

My experiments on applying Q-learning to the Gym MountainCar environment provided valuable insights into the impact of key parameters on the learning process. The findings contribute to a better understanding of how different choices of hyperparameters can influence the performance of RL algorithms in complex environments. Also this experiment showed me it is necessary to convert continious values to discrete values to reflect them to Q-Table. 

Learning rate and discount rate are critical parameters that affect convergence and overall performance in the Q-learning algorithm. Generally, the learning rate determines the extent to which the agent will update its Q values based on new information. 

As can be seen from the graphs, a high learning rate leads to faster updates, but carries the risk of exceeding the optimum values. A low learning rate causes convergence to slow down. Striking the right balance is crucial to achieving a stable and productive learning process.

The discount rate affects how the policy evaluates future rewards. We can see this better when interpreting the graphs. A higher discount rate encourages forward-thinking decision-making by emphasizing long-term rewards, while a lower discount rate encourages the policymaker to prioritize immediate rewards. 

Tuning these parameters requires careful consideration of the specific task and environment; because optimum values may vary depending on the nature of the problem at hand. Fine-tuning these parameters can significantly impact the Q-learning algorithm's ability to find optimal policies and navigate complex decision spaces. It can be seen from the graphs that the most appropriate learning rate parameter for the example in this study is 0.1. Again, when we look at the graphs, the value of 0.9 gives the most optimum result for the discount rate. 

Here is reward episode graphs under different parameters:



![Learning Rate: 0.1 Discount Rate: 0.1](/pics/lr0.1dr0.1.jpeg "Learning Rate: 0.1 Discount Rate: 0.1").

Learning Rate: 0.1 Discount Rate: 0.1

---

![Learning Rate: 0.1 Discount Rate: 0.5](/pics/lr0.1dr0.5.jpeg "Learning Rate: 0.1 Discount Rate: 0.5").

Learning Rate: 0.1 Discount Rate: 0.5

---

![Learning Rate: 0.1 Discount Rate: 0.9](/pics/lr0.1dr0.9.jpeg "Learning Rate: 0.1 Discount Rate: 0.9").

Learning Rate: 0.1 Discount Rate: 0.9

---

![Learning Rate: 0.5 Discount Rate: 0.1](/pics/lr0.5dr0.1.jpeg "Learning Rate: 0.5 Discount Rate: 0.1").

Learning Rate: 0.5 Discount Rate: 0.1

---

![Learning Rate: 0.5 Discount Rate: 0.5](/pics/lr0.5dr0.5.jpeg "Learning Rate: 0.5 Discount Rate: 0.5").

Learning Rate: 0.5 Discount Rate: 0.5

---

![Learning Rate: 0.5 Discount Rate: 0.9](/pics/lr0.5dr0.9.jpeg "Learning Rate: 0.5 Discount Rate: 0.9").

Learning Rate: 0.5 Discount Rate: 0.9

---

![Learning Rate: 0.9 Discount Rate: 0.1](/pics/lr0.9dr0.1.jpeg "Learning Rate: 0.9 Discount Rate: 0.1").

Learning Rate: 0.9 Discount Rate: 0.1

---

![Learning Rate: 0.9 Discount Rate: 0.5](/pics/lr0.9dr0.5.jpeg "Learning Rate: 0.9 Discount Rate: 0.5").

Learning Rate: 0.9 Discount Rate: 0.5

---

![Learning Rate: 0.9 Discount Rate: 0.9](/pics/lr0.9dr0.9.jpeg "Learning Rate: 0.9 Discount Rate: 0.9").

Learning Rate: 0.9 Discount Rate: 0.9

---


