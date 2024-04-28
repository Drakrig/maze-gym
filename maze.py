import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import pygame
from typing import TYPE_CHECKING, Optional, Dict, Tuple, List, Type, Any

class StateV3():
    """Maze states representation class. Store information about state coords, type, reward and neighbours.

    :param coords: coordinates of the state in the maze
    :type coords: np.ndarray
    :param valid_actions: List of avaliable and valid actions in numerical form. Used only during maze creation
    :type valid_actions: List[int]
    :param state_type: Type of the state as string, defaults to "normal"
    :type state_type: str, optional
    :ivar neighbours: Dictionary, where keys are directions in numerical form 
        and values are references to :class:`Statev3` object
    :vartype neighbours: dict
    """
    def __init__(self, coords: np.ndarray, valid_actions:List[int], state_type: str="normal"):
        """Constructor method
        """
        self._coords = coords
        self.type = state_type
        self.reward = 0
        self.valid_actions = valid_actions
        self.neighbours = dict(zip(valid_actions, [None]*len(valid_actions)))


    @property
    def coords(self) -> np.ndarray:
        """Return state coords as NumPy array.

        :return: :class:np.ndarray with state coordinates
        :rtype: np.ndarray"""
        return self._coords
    
    @property
    def observation(self) -> np.ndarray:
        """Return state existing neighbours as bool array in format [N, E, S, W]

        :return: :class:`np.ndarray` with bool value that represent in neighbour state in
            corresponded direction exists
        :rtype: np.ndarray
        """
        neighbours = [True if neighbour is not None else False for neighbour in self.neighbours.values()]
        return np.array(neighbours, dtype=np.bool_)

    @property
    def xy(self) -> Tuple[int,int]:
        """Return state coords as tuple

        :return: :class:`tuple()` of :class:`int` with state coordinates
        :rtype: _type_
        """
        return tuple(self.coords)

    def get_valid_neighbours(self) -> np.ndarray:
        """Return state neighbours

        :return: :class:`np.ndarray` with references to :class:`Statev3` objects
        :rtype: np.ndarray
        """
        neighbours = np.array((list(self.neighbours.values())))
        return neighbours[neighbours != None]
    
    def add_neighbour(self, action:int, neighbour:'StateV3') -> None:
        """Add neighbour to state

        :param action: Direction, encoded as integer
        :type action: int
        :param neighbour: Reference to :class:`Statev3` object
        :type neighbour: StateV3
        """
        self.neighbours[action] = neighbour
        self.valid_actions.remove(action)

class MazeBuilderV4():
    """Class for building maze

    :param actions: set of possible actions (action space). It should determine possible movements for agent in maze.
    Usually its movements to north, east, south and west, but it may be possible to add another directions (south-east, north-west etc.).
    Has to be a dictionary where keys are encoded direction and values are actual movements, which encodes how state coords will change 
    :type actions: Dict[int, np.ndarray]
    :param shape: determine the shape of maze. Currently may have only 2 demensions. May be assimetrical as well, for example (20,10), defaults to (10,10)
    :type shape: Tuple[int,int], optional
    :param connections_limit: Determine how much per state connections are allowed, defaults to 3
    :type connections_limit: int, optional
    :param fullfill: Determine what percent of maximum possible total connections will be used during creation with respect to max connection per state. 
    For example, for maze of size (10,10) with connection_limit = 4 max amount of connection is 180. If fullfill = 0.5 then we will have 180 * 0.5 = 90 connections in maze.
    Defaults to 0.5
    :type fullfill: float, optional
    :param seed: random seed for creation, so we can control the process, defaults to None
    :type seed: int, optional
    """
    def __init__(self, 
                 actions:Dict[int, np.ndarray], 
                 shape: Tuple[int,int]=(10,10), 
                 connections_limit:int=3, 
                 fullfill:float=0.5, 
                 seed: int=None):
        """Constructor method
        """
        self.movements = actions
        self.shape = shape
        self.states = {}
        self.connections_limit = connections_limit
        self.fullfill = fullfill
        self.seed = seed if seed is not None else None
        self.total_pathes = self.shape[0] * (self.shape[1] - 1) + self.shape[1] * (self.shape[0] - 1)
        if self.connections_limit < len(shape) * 2:
            #Discount factor for less connections
            self.total_pathes = self.total_pathes - int(self.total_pathes * 0.16)
        np.random.seed(self.seed)
        x_entrances = np.arange(1, self.shape[0]+1)
        y_entrances = np.arange(1, self.shape[1]+1)

        grid = np.array(np.meshgrid(x_entrances, y_entrances)).T.reshape(-1, 2)
        mask = np.apply_along_axis(lambda x:(x[0] == 1 or x[0] == self.shape[0]) or \
                                            (x[1] == 1 or x[1] == self.shape[1]), arr=grid, axis=1)
        
        possible_entrance = grid[mask]
        grid = grid.reshape(self.shape[0],self.shape[1],2)

        self.entrance = possible_entrance[np.random.choice(possible_entrance.shape[0])]

        state = self._create_state(self.entrance)
        self.states[state.xy] = state
        self.current_state = self.states[state.xy]

        self.filled_pathes=0
        self._create_maze()
    
    def _is_full_enoght(self) -> bool:
        """Check main loop stop condition - if enough pathes were created

        :return: True if stop condition reached, False otherwise
        :rtype: bool
        """
        return self.filled_pathes < self.total_pathes * self.fullfill

    #State Section

    def _create_state(self, coords:np.ndarray) -> Type[StateV3]:
        """Create new state

        :param coords: coordinates of the newly created states
        :type coords: np.ndarray
        :return: New :class:`state.StateV3` object
        :rtype: Type[StateV3]
        """
        actions = self._find_valid_action(coords)
        return StateV3(coords, actions)
    
    def _if_state_exist(self, coords:Tuple[int,int]) -> bool:
        """Check if state already exists in maze

        :param coords: Coords of the states in tuple form
        :type coords: Tuple[int,int]
        :return: True if state was already created, False oterwise
        :rtype: bool
        """
        try:
            self.states[coords]
            return True
        except:
            return False
    
    def _find_valid_action(self, coords:np.ndarray) -> Tuple[Dict, List[int]]:
        """Find action, valid for coordinates

        :param coords: _description_
        :type coords: np.ndarray
        :return: _description_
        :rtype: Tuple[Dict, List[int]]
        """
        valid_actions = list()
        for key in self.movements.keys():
            if (np.zeros(2) < self.movements[key] + coords).all() \
                & (self.movements[key] + coords <= np.array(self.shape)).all():
                valid_actions.append(key)
        return valid_actions

    #Action Section
    
    def _opposite_direction(self, action:int) -> int:
        """Find opposite direction as code value. For example, for north will return south

        :param action: Selected action
        :type action: int
        :return: Action that leads to opposite direction
        :rtype: int
        """
        return (action + len(self.movements)/2) % len(self.movements)
    
    # Neighbours Section
    
    def _connect_states(self,action:int, state:Type[StateV3], next_state:Type[StateV3]) -> None:
        """Create connection between two states by adding referencecs in each other neighbours dictionary

        :param action: Encoded action value
        :type action: int
        :param state: One state
        :type state: Type[StateV3]
        :param next_state: Another state
        :type next_state: Type[StateV3]
        """
        state.add_neighbour(action, next_state)
        opposite = int(self._opposite_direction(action))
        next_state.add_neighbour(opposite, state)
    
    def _create_maze(self) -> None:
        """Class main loop"""
        limit = 10000
        self._step = 0
        while self._is_full_enoght() and self._step <= limit:
            self._create_path(self.current_state)
            if self.seed:
                self.seed+=1
    
    def _create_path(self, state:Type[StateV3]) -> None:
        """Main creation method. Do the following:

        1. Check if reached limit for per state connections or any valid action left.
          - if any true, change current state to one of its neighbours and goes to next main loop step
        2. Take one of the avaliable valid actions and check if any state exist in such coordinates
          - if no state exist, create new state and create connection between new and current state
          - if state exist, check if it has reached limir for per state number of cinnection
            - if limit is reached, remove selected earlier action from valid ones and trying to select another action
            - otherwise create connection between two states and change current state to selected neighbour state

        :param state: Current state
        :type state: Type[StateV3]
        """
        if self.seed:
            np.random.seed(self.seed)
        if state.get_valid_neighbours().shape[0] == self.connections_limit\
            or len(state.valid_actions) == 0:
            neighbours = state.get_valid_neighbours()
            idx = np.random.choice(neighbours.shape[0])
            next_state = neighbours[idx]
            self._step+=1
            self.current_state = next_state
            return
        while len(state.valid_actions) != 0:
            idx = np.random.choice(len(state.valid_actions))
            action = self.movements[state.valid_actions[idx]]
            next_state_pos = state.coords + action
            try:
                next_state = self.states[tuple(next_state_pos)]
                next_neighbours = next_state.get_valid_neighbours()
                if next_neighbours.shape[0] == self.connections_limit:
                    state.valid_actions.remove(state.valid_actions[idx])
                    self._step+=1
                    continue
                self._connect_states(state.valid_actions[idx], state, next_state)
                self.filled_pathes+=1
                self._step=0
            except KeyError:
                next_state = self._create_state(next_state_pos)
                self._step=0
                self.states[next_state.xy] = next_state
                self._connect_states(state.valid_actions[idx], state, next_state)
                self.filled_pathes+=1
            finally:
                self.current_state = next_state
                return


class MazeV5(gym.Env):
    """Maze enviroment for path finding task

    :param shape: Shape of the maze, defaults to (10,10)
    :type shape: Tuple[int,int] | None, optional
    :param connections_limit: Determine how much per state connections are allowed, defaults to 3
    :type connections_limit: int | None, optional
    :param fullfill: Determine what percent of maximum possible total connections will be used during creation with respect to max connection per state.
    For example, for maze of size (10,10) with connection_limit = 4 max amount of connection is 180. 
    If fullfill = 0.5 then we will have 180 * 0.5 = 90 connections in maze. Defaults to 0.5
    :type fullfill: float | None, optional
    :param traps_percent: Percent of the states to set as traps. Currently unused since requere more balancing both reward and observation space, defaults to 0
    :type traps_percent: float | None, optional
    :param distance_quantile: Using to select goal state. Selection based on distance from maze entrance (starting point), defaults to 0.75
    :type distance_quantile: float | None, optional
    :param dynamic_reward: Flag for selection type of reward for step, defaults to True
    :type dynamic_reward: bool | None, optional
    :param seed: Random seed for maze creation, so we can control the process, defaults to None
    :type seed: int | None, optional
    :param render: Render flag. If True, also requare to select render mode, defaults to False
    :type render: bool | None, optional
    :param render_mode: Determine how to handle rendered maze, possible values [human, rgb_array], defaults to None
    :type render_mode: str | None, optional 
    """
    def __init__(self,
                shape: Tuple[int,int]|None = (10,10),
                connections_limit: int|None = 3,
                fullfill: float|None = 0.5,
                traps_percent: float|None = 0,
                distance_quantile: float|None = 0.75,
                dynamic_reward: bool|None = True,
                seed: int|None = None,
                render: bool|None = False,
                render_mode: str|None = None):
        """Constructor method
        """
        super(MazeV5, self).__init__()
        self.movements = {0:np.array([0,1]), 1:np.array([1,0]), 2:np.array([0,-1]), 3:np.array([-1,0])}
        builder = MazeBuilderV4(self.movements, shape, connections_limit, fullfill, seed)
        self.states = builder.states.copy()
        self.action_space = spaces.Discrete(len(self.movements))
        self.entrance = builder.entrance.copy()
        del(builder)
        self.current_state = self.states[tuple(self.entrance)]
        self.min_distance_quant = distance_quantile
        self.seed = seed if seed is not None else None
        self.goal = None
        self.total_states = len(self.states)
        self.shape = shape
        self._repeat = 1
        self.dynamic_reward=dynamic_reward
        self.max_distance = self._calc_distance(np.array([1,1]), np.array(shape))
        self._place_special_points(traps_percent)
        self.observation_space = spaces.Box(shape=self.get_observation().shape, low=0, high=1)
        #Render
        self.use_render = render
        if self.use_render:
            pygame.init()
            self._render_state_size=10
            self._render_connection_length=np.array([90,90])
            self._colors = {"normal":"blue", "treasure":"yellow", "agent":"red"}
            self._font = pygame.font.SysFont('couriernew', 18)
            self.compass_center = np.array([540, 100])
            self._compass_line_len = np.sqrt(np.power(np.array([640,95]) - self.compass_center, 2).sum())
            self._render_deep = 6
            self._render_mode = render_mode
            self.surface = pygame.Surface((640, 640))
            if render_mode == "human":
                self.screen = pygame.display.set_mode((640, 640))
            self.render()
    
    def calc_direction(self) -> np.ndarray: #v5
        """Method for calculation of angle between X and Y axes and vector that points from current state to goal.
        Value normalized between 0 and 180
        Should (probably) work as compass for agent.

        :return: Normalized angel between dimentional axes and directional vector
        :rtype: np.ndarray
        """
        ax, ay = np.array([0, 1]), np.array([1, 0])
        vector = self.goal - self.current_state.coords
        if vector.sum()==0:
            return np.zeros(2)
        cos_x = np.dot(ax, vector) / (np.linalg.norm(ax) * np.linalg.norm(vector))
        cos_y = np.dot(ay, vector) / (np.linalg.norm(ay) * np.linalg.norm(vector))
        angle_degrees = np.array([np.degrees(np.arccos(cos_x)),
                                np.degrees(np.arccos(cos_y))])
        return angle_degrees / 180

    def _calc_step_reward(self) -> float:
        """Calculate step reward by following formula
        -0.01 * normalized_eucledean_distance_to_goal
        So, reward will become bigger, as we approach goal state.
        By appling normalization we make reward more universal for all kind of shapes and distances

        :return: _description_
        :rtype: float
        """
        distance_reward = -0.01 * (self._calc_distance(self.goal, self.current_state.coords) / self.max_distance)
        return distance_reward + self.current_state.reward
    
    def _calc_distance(self, point1:np.ndarray, point2:np.ndarray) -> float:
        """Calculate eucledean distance between 2 coordinates

        :param point1: One coordinate
        :type point1: np.ndarray
        :param point2: Another coordinate
        :type point2: np.ndarray
        :return: Eucledean distance between coordinates
        :rtype: float
        """
        return np.sqrt(np.power(point1 - point2, 2).sum())

    def _is_valid_action(self, action:np.ndarray) -> bool:
        """Check if selected action is valid for current state 

        :param action: _description_
        :type action: in
        :return: _description_
        :rtype: bool
        """
        possible_coords=[]
        for key in self.current_state.neighbours.keys():
            if self.current_state.neighbours[key] is not None:
                possible_coords.append(self.current_state.neighbours[key].xy)
        new_coords = tuple(self.current_state.coords + self.movements[action])
        return new_coords in possible_coords
    
    def _place_goal(self, points:np.ndarray) -> None:
        """Selects state to make it a goal state. For that uses eucledean distance between starting state and all another states.
        Then filter it by specified quantile and randomly select one of them.

        :param points: Array with states coordinates
        :type points: np.ndarray
        """
        distances = np.apply_along_axis(self._calc_distance, 
                                        arr=points,
                                        point2=self.current_state.coords,
                                        axis=1)
        limit = np.quantile(distances, self.min_distance_quant)
        distances = distances.reshape(points.shape[0],1) 
        idxs = np.arange(0, points.shape[0]).reshape(points.shape[0], 1)
        df = np.hstack([idxs, distances]).reshape(distances.shape[0],2,1)  
        mask = (df[:,1,:] >= limit).reshape(points.shape[0])
        df = df[mask]
        if self.seed:
            np.random.seed(self.seed)
        idx = df[np.random.randint(0, df.shape[0])][0][0].astype(int)
        self.goal = points[idx]
        self.states[tuple(self.goal)].type = "treasure"
        self.states[tuple(self.goal)].reward = self._calc_distance(self.entrance, self.goal) if self.dynamic_reward else self.shape[0]*self.shape[1]

    def _place_special_points(self, traps_percent:float) -> None:
        """Place traps and goal in maze

        :param traps_percent: Percentage of states that will be traps
        :type traps_percent: float
        """
        points = list(self.states.keys())
        points.remove(self.current_state.xy)
        limit = np.round(len(points) * traps_percent)
        counter = 0
        while counter < limit:
            idx = np.random.randint(0, len(points))
            if self.states[points[idx]].type == "normal":
                self.states[points[idx]].type = "trap"
                self.states[points[idx]].reward = -(0.1*self.shape[0]*self.shape[1]) **\
                                                   ((1 - traps_percent) * (self.total_states / (self.shape[0]*self.shape[1])))
                counter += 1
        if self.goal is None:
            self._place_goal(np.array(points))

    def get_observation(self) -> np.ndarray:
        """Function for returning current observation of environment.
        Separated from step method, so can be modified if necessary.

        :return: Observation as :class:`np.ndarray`
        :rtype: np.ndarray
        """
        distance = self._calc_distance(self.current_state.coords, self.goal) / self.max_distance
        angel = self.calc_direction() #v5
        observation = np.append(np.append(distance, angel), self.current_state.observation)
        return observation
    
    def reset(self, seed:int | None = None, options: dict[str, Any] | None = None) -> Tuple[np.ndarray, dict[str, Any]]:
        """Standrat reset method

        :param seed: selected random seed, defaults to None
        :type seed: int | None, optional
        :param options: Additional options for enviromnment. Added for compability, defaults to None
        :type options: dict[str, Any] | None, optional
        :return: Observation and environment info 
        :rtype: Tuple[np.ndarray, dict[str, Any]]
        """
        self.current_state = self.states[tuple(self.entrance)]
        if self.use_render:
            self.render()
        return self.get_observation(), {}
    
    def step(self, action:int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step method of environment

        :param action: Action, selected by agent
        :type action: int
        :return: Environment observation, step reward, done flag, False(whatever is it), and enviroment info 
        :rtype: Tuple[np.ndarray, float, bool, bool, Dict]
        """
        # Take a step with the given action and return the next state, reward, done, and info
        valid_action = self._is_valid_action(action)
        if valid_action:
            self._repeat=1
            new_state = tuple(self.current_state.coords + self.movements[action])
            self.current_state = self.states[new_state]
            reward = self._calc_step_reward() if self.dynamic_reward else -0.1
        else:
            self._repeat+=1
            reward = self._calc_step_reward() * (2**self._repeat) if self.dynamic_reward else -1
        done = all(self.current_state.coords == self.goal)
        observation = self.get_observation()
        return observation, reward, done, False, dict()

    def _draw_state_neighbours(self, state:StateV3, center:np.ndarray, deep:int) -> None:
        """Recursive function for rendering neighbouring states

        :param state: State
        :type state: StateV3
        :param center: Relative center from which lines will be drawn
        :type center: np.ndarray
        :param deep: Stop flag for rendering
        :type deep: int
        """
        if deep >= self._render_deep:
            return
        for neighbour in state.neighbours.values():
            if neighbour is not None:
                direction = (state.coords - neighbour.coords) * np.array([-1,-1])
                coords = center + self._render_connection_length*(direction)
                pygame.draw.lines(self.surface, "purple",True, [center,coords], 4)
                pygame.draw.circle(self.surface, self._colors[neighbour.type] ,coords, self._render_state_size)
                self._draw_state_neighbours(neighbour, coords, deep+1)
    
    def render(self) -> np.ndarray|None:
        """Renders enviroment graphical representation with PyGame with 640x640 resolution.
        Since maze may have different size, renders only nearies states.

        :return: NumPy array with rendered pixels or None if for-human rendering is selected
        :rtype: np.ndarray|None
        """
        #Main screen
        self.surface.fill("grey")
        self._draw_state_neighbours(self.current_state, np.array([320,320]),0)
        pygame.draw.circle(self.surface,self._colors["agent"], (320,320), self._render_state_size)
        text = self._font.render(str(f'Agent {self.current_state.xy}'), True, "white")
        self.surface.blit(text, np.array([330,260]) + np.array([0,20]))
        #Compass
        pygame.draw.polygon(self.surface, "grey", [(640,0),(440,0),(440,240), (640,240)])
        pygame.draw.polygon(self.surface, "yellow", [(640,0),(440,0),(440,240), (640,240)], 4)
        pygame.draw.lines(self.surface, "yellow", True,[(440,200), (640,200)], 4)
        text = self._font.render(str(f'Goal {self.goal}'), True, "yellow")
        direction = (self.goal-self.current_state.coords)
        v_s = self._compass_line_len / np.sqrt(np.power(direction, 2).sum())
        pointer = self.compass_center + direction * v_s
        pygame.draw.lines(self.surface, "yellow",True, [self.compass_center,pointer], 4)
        pygame.draw.circle(self.surface, "yellow" ,self.compass_center, self._render_state_size)
        self.surface.blit(text, ((465,210)))
        if self._render_mode == "human":
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()
        elif self._render_mode == "rgb_array":
            return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.surface)), axes=(1, 0, 2)
                )

    def _plot_maze(self, figsize: Tuple[int, int] = (5,5), save: bool|None = False) -> None:
        """Plot maze map with matplolib. Allow to see all maze at once.

        :param figsize: size of the plot, defaults to (5,5)
        :type figsize: Tuple[int, int], optional
        :param save: if True save plot as png image in working directory, defaults to False
        :type save: bool | None, optional
        """
        plt.figure(figsize=figsize)

        for state in self.states.values():
            for neighbour in state.neighbours.values():
                if neighbour is not None:
                    plt.plot((state.coords[1], neighbour.coords[1]), (state.coords[0], neighbour.coords[0]), c="red")
            plt.scatter(x=state.coords[1], y=state.coords[0], c="blue" if state.type == "normal" else "red")
        plt.scatter(x=[self.entrance[1]], y=[self.entrance[0]], c="lightgreen")
        plt.scatter(x=[self.goal[1]], y=[self.goal[0]], c="yellow")
        if save:
            plt.savefig('my_plot.png', format='png', dpi=300)