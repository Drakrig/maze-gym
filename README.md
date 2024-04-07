# Maze pathfinding enviroment
## Description

Maze enviroment consist a set of connected states. May be also describe as undirected simple graph. The goal is to find exit or treasure by moving from starting position to it.

## Main features

* Customazible maze creation - you may change shape, limit connections per state and influence distance to goal.
* Creation control with random seed. Builder logic describe in builder.md
* 2 types of reward avaliable - static values and dynamicly calculated.
* Based on gymnasium so can be used with RL framework with minimal changes.

## Observation space

Currently, observation space is an array array with 7 elements:

1. Normalized eucledean distance between the current state and the goal state. For normalization used distance between point (1,1) and (maze_shape[0],maze_shape[1]) (because state coords starts from 1).

2. Normalized angel in degrees between vector that points from current state to goal and X axe. Normalized by dividing on 180. Should work like compass.

3. Normalized angel in degrees between vector that points from current state to goal and Y axe. Normalized by dividing on 180. Should work like compass.

4. to 7. Bool values that represent if there is path to respective direction. Curretnly available directions are N, E, S, W.

## Step reward

There are 2 types of reward that can be selected:

1. Static reward on each step return -0.1 for valid action, -1 for selecting wrong action (walk to wall) and product of maze shape as reward for reaching goal.

2. Dynamic reward use more complex reward: 

* -1% of current normalized eucledean distance as step reward for selecting valid action
* For selesting invalid action we return `-0.01*normalized eucledean distance * (2 ** repeat)` where repeat is a number of times wrong actions were selected repeatedly. So, with each wrong selection we punish agent harder and harder.
* Eucledean distance between starting point and goal as reward for completing maze.

Motivation behind this particular way to calculate reward is to give algorithm additional hints during optimization steps so it'll more value trajectories that leads towards goal. We simply compensate (to some extend) discount factor for closing to goal and emphasize penalty when we goes to opposite.

In table below some calculations for discounted reward with 3 steps and gamma=0.995.

|                          | Reward (10x10) | Reward (50x50) | Reward (100x100) |
|--------------------------|----------------|----------------|------------------|
| Moving away (dynamic)    | -0.00766       | -0.002288      | -0.001393        |
| Moving towards (dynamic) | -0.003687      | -0.001295      | -0.000897        |
| Moving (static)          | -0.002985      | -0.002985      | -0.002985        |

As you can see, there is some drawbacks as maze become bigger - rewards difference is less noticable but still exists.

By default dynamic reward is used. Still, the way reward is been calculated may be changed in future to make it more simple and clear.

## Notes about time creation and render

Since maze could be pretty huge, creation time may vary and depends on shape and fillness. Usually it's matter of seconds for shape up to (100,100) and may increse to minutes in case of (1000+,1000+).

Because of the mazes size, render is also limited by depth, so we render only observable fraction of the maze, since its hard to render big mazes in reasanoble form.