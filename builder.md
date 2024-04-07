# MazeBuilder

Maze builder create unique mazes due to its random nature. In the same time its 100% contollable with seed.

## Builder logic
### Class arguments

Class `MazeBuilderV4` recive the following arguments:

* `actions`: set of possible actions (action space). It should determine possible movements for agent in maze. Usually its movements to norths, east, south and west, but it may be possible to add another directions (south-east, north-west etc.) 
* `shape`: default `(10,10)` determine the shape of maze. Currently may have only 2 demensions. May be assimetrical as well, for example (20,10) 
* `connections_limit`: default `3`. Determine how much per state connections are allowed
* `fullfill`: default `0.5`. Determine what percent of maximum possible total connections will be used during creation with respect to max connection per state. For example, for maze of size `(10,10)` with `connection_limit` = 4 max amount of connection is `180`. If `fullfill` = `0.5` then we will have `180 * 0.5 = 90` connections in maze. 
* `seed`: random seed for creation, so we can control the process

### Initial steps

First step is to determine entrance of the maze. It's starting point for agent. For that we create arrays wit possible values of X and Y (we using value from 1 to shape[X|Y], only reason so we can draw pretty plot with matplotlib). After that we create grid and use mask to select border values. Then just randomly select one of them.

### Main loop

We using standart dictionary to save our maze. Keys are states coordinates and value are states. For control we use couple variables:

* `filled_pathes` to save total amount of created pathes
* `total_pathes` to save maximum amount of pathes (with respect to max per state connections)
* `limit` to handle situation when we can't find a valid step and just walking around maze without creation of new states and connections. Currenly it's  limited to 1000 steps.
* `step` to count unsufficinet steps

So, we'll continue creation process until we create enought passes (`filled_pathes == total_pathes * fullfill`) or while not stack for `limit` steps.

After each main loop step, we increase seed value so we don't stack.

#### Path creation

We start at initial entrance state that was created in initialization phase. state.get_valid_neighbours().shape[0] == self.connections_limit

Firstly, we chould check 2 things:

1. Is current state still have valid amount connections left (`state.get_valid_neighbours().shape[0] == self.connections_limit`)

2. Is current state have any valid actions left. Action is unvalid if it leads to left maze, for example, if after that agent will appear in state (11,1), but maze shape is (10,10).

If any of this is true then we should step into one of the existing neigbour states by randomly selecting one of then.

Otherwise, we randomly choose one of the available valid action. Here is 2 possibilities:

1. The next state is not exist. Then we just create new state and connect both current and new states.

2. The next state is exist and :
    
    - it have reached its connection limit. In that case we remove selected action from valid, increase unsufficient state counter and goes to next circle step

    - it still have available connections. So we connect them.

In the end, we change current state to created or selected one.

