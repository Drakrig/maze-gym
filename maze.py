import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import pygame
from typing import TYPE_CHECKING, Optional

class StateV3():
    """
    Maze states representation class. Store information about state coords, type, reward and neighobiurs
    """
    def __init__(self, coords, neighbours, valid_actions, state_type="normal"):
        self._coords = np.array(coords)
        self.type = state_type
        self.reward = 0
        self.neighbours = neighbours
        self.valid_actions = valid_actions

    @property
    def coords(self):
        """Return state coords as NumPy array"""
        return self._coords
    
    @property
    def observation(self):
        """Return state existing neighbours as bool array in format [N, E, S, W]"""
        neighbours = [True if neighbour is not None else False for neighbour in self.neighbours.values()]
        return np.array(neighbours, dtype=np.bool_)

    @property
    def xy(self):
        """Return state coords as tuple"""
        return tuple(self.coords)

    def get_valid_neighbours(self):
        """Return state neighbours"""
        neighbours = np.array((list(self.neighbours.values())))
        return neighbours[neighbours != None]
    
    def add_neighbour(self, action, neighbour):
        """Add neighbour to state"""
        self.neighbours[action] = neighbour
        self.valid_actions.remove(action)

class MazeBuilderV4():
    """Class for building maze
    Args:
* actions: set of possible actions (action space). It should determine possible movements for agent in maze. Usually its movements to norths, east, south and west, but it may be possible to add another directions (south-east, north-west etc.) 
* shape: default (10,10) determine the shape of maze. Currently may have only 2 demensions. May be assimetrical as well, for example (20,10) 
* connections_limit: default 3. Determine how much per state connections are allowed
* fullfill: default 0.5. Determine what percent of maximum possible total connections will be used during creation with respect to max connection per state. For example, for maze of size (10,10) with connection_limit = 4 max amount of connection is 180. If fullfill = 0.5 then we will have 180 * 0.5 = 90 connections in maze. 
* seed: random seed for creation, so we can control the process
    """
    def __init__(self, actions, shape=(10,10), connections_limit=3, fullfill=0.5,seed=None):
        self.movements = actions
        self.shape = shape
        self.size = shape[0] * shape[1]
        self.states = {}
        self.connections_limit = connections_limit
        self.fullfill = fullfill
        self.seed = seed if seed is not None else None
        self.total_x = shape[0]
        self.total_y = shape[1]
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
        self.grid = grid.reshape(self.shape[0],self.shape[1],2)

        self.entrance = possible_entrance[np.random.choice(possible_entrance.shape[0])]

        state = self._create_state(self.entrance)
        self.states[state.xy] = state
        self.current_state = self.states[state.xy]

        self.filled_pathes=0
        self._create_maze()
    
    def _is_full_enoght(self):
        return self.filled_pathes < self.total_pathes * self.fullfill

    #State Section

    def _create_state(self, coords):
        neighbours, actions = self._find_neighbours(coords)
        return StateV3(coords, neighbours, actions)
    
    def _if_state_exist(self, coords):
        try:
            self.states[coords]
            return True
        except:
            return False

    #Action Section
    
    def _if_valid_action(self, coords, action):
        next_coords = coords + action
        for i in range(len(self.shape)):
            if not 0<next_coords[i]<=self.shape[i]:
                return False
        return True
    
    def _opposite_direction(self, action):
        return (action + len(self.movements)/2) % len(self.movements)
    
    # Neighbours Section

    def _find_neighbours(self, coords):
        valid_directions = dict()
        valid_actions = list()
        for key, action in self.movements.items():
            next_coords = np.array(coords) + action
            valid = self._if_valid_action(coords, action)
            valid_directions[key] = None
            if valid:
                valid_actions.append(key)
        return valid_directions, valid_actions
    
    def _connect_states(self,action, state, next_state):
        state.add_neighbour(action, next_state)
        opposite = int(self._opposite_direction(action))
        next_state.add_neighbour(opposite, state)
    
    def _create_maze(self):
        """Class main loop"""
        limit = 10000
        self._step = 0
        while self._is_full_enoght() and self._step <= limit:
            self._create_path(self.current_state)
            if self.seed:
                self.seed+=1
    
    def _create_path(self, state):
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
    Args:
* shape (int,int): default (10,10) shape of the maze
* connections_limit int: default 3. Determine how much per state connections are allowed
* fullfill float: default 0.5. Determine what percent of maximum possible total connections will be used during creation with respect to max connection per state. For example, for maze of size (10,10) with connection_limit = 4 max amount of connection is 180. If fullfill = 0.5 then we will have 180 * 0.5 = 90 connections in maze. 
* traps_percent float: default 0. Percent of the states to set as traps. Currently unused since requere more balancing both reward and observation space
* distance_quantile float: default 0.75. Using to select goal state. Selection based on distance from maze entrance (starting point)
* dynamic_reward bool: default True. Set type of reward. 
* seed int: random seed for creation, so we can control the process
* render bool: default False. Render flag. If True, also requare to select render mode
* render_mode str: optional, possible values [human, rgb_array]. Determine how to handle rendered maze
    """
    def __init__(self,
                shape=(10,10),
                connections_limit=3,
                fullfill=0.5,
                traps_percent=0,
                distance_quantile=0.75,
                dynamic_reward=True,
                seed=None,
                render=False,
                render_mode: Optional[str] = None):
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
    
    def calc_direction(self): #v5
            ax, ay = np.array([0, 1]), np.array([1, 0])
            vector = self.goal - self.current_state.coords
            if vector.sum()==0:
                return np.zeros(2)
            cos_x = np.dot(ax, vector) / (np.linalg.norm(ax) * np.linalg.norm(vector))
            cos_y = np.dot(ay, vector) / (np.linalg.norm(ay) * np.linalg.norm(vector))
            angle_degrees = np.array([np.degrees(np.arccos(cos_x)),
                                    np.degrees(np.arccos(cos_y))])
            return angle_degrees / 180

    def _calc_step_reward(self):
        distance_reward = -0.01 * (self._calc_distance(self.goal, self.current_state.coords) / self.max_distance)
        return distance_reward + self.current_state.reward
    
    def _calc_distance(self, point1, point2):
        return np.sqrt(np.power(point1 - point2, 2).sum())

    def _is_valid_action(self, movement):
        possible_coords=[]
        for key in self.current_state.neighbours.keys():
            if self.current_state.neighbours[key] is not None:
                possible_coords.append(self.current_state.neighbours[key].xy)
        new_coords = tuple(self.current_state.coords + movement)
        return new_coords in possible_coords
    
    def _place_goal(self, points):
        distances, pathes = np.array([]), np.zeros((1,2))
        for i in range(points.shape[0]):
            distance = self._calc_distance(self.entrance, points[i])
            distances = np.append(distances, distance)
            pathes = np.vstack((pathes, points[i]))
        limit = np.quantile(distances, self.min_distance_quant)
        mask = distances >= limit
        pathes = pathes[1:]
        pathes = pathes[mask]
        np.random.seed(self.seed)
        if self.seed:
            np.random.seed(self.seed)
        idx = np.random.randint(0, pathes.shape[0])
        self.goal = pathes[idx]
        self.states[tuple(self.goal)].type = "treasure"
        self.states[tuple(self.goal)].reward = self._calc_distance(self.entrance, self.goal) if self.dynamic_reward else self.shape[0]*self.shape[1]

    def _place_special_points(self, traps_percent):
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

    def get_observation(self):
        distance = self._calc_distance(self.current_state.coords, self.goal) / self.max_distance
        angel = self.calc_direction() #v5
        observation = np.append(np.append(distance, angel), self.current_state.observation)
        return observation
    
    def reset(self, seed=None, options=None):
        self.current_state = self.states[tuple(self.entrance)]
        if self.use_render:
            self.render()
        return self.get_observation(), {}
    
    def step(self, action):
        # Take a step with the given action and return the next state, reward, done, and info
        valid_action = self._is_valid_action(self.movements[action])
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
        return observation, reward, done, False,dict()

    def _draw_state_neighbours(self, state, center, deep):
        if deep >= self._render_deep:
            return
        for neighbour in state.neighbours.values():
            if neighbour is not None:
                direction = (state.coords - neighbour.coords) * np.array([-1,-1])
                coords = center + self._render_connection_length*(direction)
                pygame.draw.lines(self.surface, "purple",True, [center,coords], 4)
                pygame.draw.circle(self.surface, self._colors[neighbour.type] ,coords, self._render_state_size)
                self._draw_state_neighbours(neighbour, coords, deep+1)
    
    def render(self):
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

    def _plot_maze(self, figsize=(5,5), save=False):
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