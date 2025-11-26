# Q-learning ROS2 Project

A reinforcement learning project where a drone learns to navigate a 4×4 maze using the Q-learning algorithm. The agent first trains inside a simulated grid-world, builds an optimal Q-table, and then the learned policy is deployed in a ROS (or ROS2-style) environment to control a drone.



## Overview

This project implements Q-learning to solve a maze navigation task with three obstacles.
After training, the drone moves autonomously through the maze following the optimal action sequence produced by the Q-table.

The project contains two main components:

1. **q_learning.py**

   * Runs Q-learning inside a Python simulation.
   * Builds and trains the Q-table.
   * Demonstrates the learned optimal route.

2. **drone_fly_qlearn.py**

   * Loads the learned policy.
   * Converts maze actions into drone motion commands.
   * Controls the drone through ROS topics.


## Demo


https://github.com/user-attachments/assets/d7b21651-ad68-4978-ab96-7cb47c21e62b



## Environment Description

The maze is a 4×4 grid:

* Start position: `(3, 0)`
* Goal position: `(0, 3)`
* Holes (termination with negative reward):

  * `(1, 3)`
  * `(1, 1)`
  * `(2, 1)`

The agent receives:

* Reward `+1` for reaching the goal
* Reward `-1` for falling into a hole
* Reward `0` otherwise

Actions: `up`, `down`, `left`, `right`



## Q-learning Hyperparameters:

* Learning rate α = 0.01
* Discount factor γ = 0.9
* Exploration rate ε = 0.9
* Episodes = 200–400 (adjustable)

After training, the optimal action sequence typically converges to:

```
right → right → up → up → up → right
```



## ROS Drone Control

The `drone_fly_qlearn.py` script:

* Starts a ROS node
* Takes off the drone
* Converts the action sequence into turning and forward motions
* Publishes to:

  * `/cmd_vel`
  * `/drone/takeoff`
  * `/drone/land`

Motion translation:

* `up`, `down`, `left`, `right` → internal numeric actions
* Converted again to yaw rotations and forward movements



## How to Run (Simulation Only)

Inside the project directory:

```
python q_learning.py
```

This will:

1. Train the agent
2. Print the final Q-table
3. Show the resulting action path



## How to Run in ROS

Inside your catkin workspace:

```
cd ~/catkin_ws
catkin_make
source devel/setup.bash
rosrun rl_project drone_fly_qlearn.py
```

The drone will:

1. Take off
2. Follow the learned Q-table path
3. Land automatically



## Project Structure

```
rl-qlearning-ros2-project/
│
├── q_learning.py
├── scripts/
│     └── drone_fly_qlearn.py
├── package.xml
├── CMakeLists.txt
└── README.md
```
