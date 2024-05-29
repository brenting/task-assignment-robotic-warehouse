import functools
from collections import Counter, OrderedDict, defaultdict
from enum import IntEnum
from typing import Any

import networkx as nx
import numpy as np
import pyastar2d
from gymnasium import spaces
from numpy.random import default_rng
from pettingzoo import ParallelEnv

from tarware.utils import find_sections

_COLLISION_LAYERS = 4

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1
_LAYER_CARRIED_SHELFS = 2
_LAYER_PICKERS = 3


class AgentType(IntEnum):
    AGV = 0
    PICKER = 1
    AGENT = 2

class Action(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    TOGGLE_LOAD = 4

class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

DIR_TO_ENUM = {
    (0, -1): Direction.UP,
    (0, 1): Direction.DOWN,
    (-1, 0): Direction.LEFT,
    (1, 0): Direction.RIGHT,
    }

def get_next_micro_action(agent_x, agent_y, agent_direction, target):
    target_x, target_y = target
    target_direction =  DIR_TO_ENUM[(target_x - agent_x, target_y - agent_y)]

    turn_order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    # Find the indices of the source and target directions in the turn order
    source_index = turn_order.index(agent_direction)
    target_index = turn_order.index(target_direction)

    # Calculate the difference in indices to determine the number of turns needed
    turn_difference = (source_index - target_index) % len(turn_order)

    # Determine the direction of the best next turn
    if turn_difference == 0:
        return Action.FORWARD
    elif turn_difference == 1:
        return Action.LEFT
    elif turn_difference == 2:
        return Action.RIGHT
    elif turn_difference == 3:
        return Action.RIGHT

class RewardType(IntEnum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2

class ImageLayer(IntEnum):
    """
    Input layers of image-style observations
    """
    SHELVES = 0 # binary layer indicating shelves (also indicates carried shelves)
    REQUESTS = 1 # binary layer indicating requested shelves
    AGENTS = 2 # binary layer indicating agents in the environment (no way to distinguish agents)
    AGENT_DIRECTION = 3 # layer indicating agent directions as int (see Direction enum + 1 for values)
    AGENT_LOAD = 4 # binary layer indicating agents with load
    GOALS = 5 # binary layer indicating goal/ delivery locations
    ACCESSIBLE = 6 # binary layer indicating accessible cells (all but occupied cells/ out of map)
    PICKERS = 7 # binary layer indicating agents in the environment which only can_load
    PICKERS_DIRECTION = 8 # layer indicating agent directions as int (see Direction enum + 1 for values)

class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y


class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, msg_bits: int, agent_type: AgentType):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.message = np.zeros(msg_bits)
        self.req_action: Action | None = None
        self.carrying_shelf: Shelf | None = None
        self.canceled_action = None
        self.has_delivered = False
        self.to_deliver = False
        self.path = None
        self.busy = False
        self.fixing_clash = 0
        self.type = agent_type

    @property
    def collision_layers(self):
        if self.loaded:
            return (_LAYER_AGENTS)
        else:
            return (_LAYER_AGENTS)

    def req_location(self, grid_size) -> tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.x, self.y
        elif self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)
        elif self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y
        elif self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir


class Shelf(Entity):
    counter = 0

    def __init__(self, x, y):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)

    @property
    def collision_layers(self):
        return ()

class Warehouse(ParallelEnv):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        n_agvs: int,
        n_pickers: int,
        msg_bits: int,
        sensor_range: int,
        request_queue_size: int,
        max_inactivity_steps: int | None,
        max_steps: int | None,
        reward_type: RewardType,
        layout: str | None = None,
        observation_type: str = "flattened",
        normalised_coordinates: bool = False,
        render_mode: str | None = None,
        action_masking_level: int = 2,
        sample_collection: str = "all",
        targets_vam: bool = True,
        agents_can_clash: bool = True,
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where agents are rewarded if
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param n_agvs: Number of spawned and controlled agv
        :type n_agvs: int
        :param n_pickers: Number of spawned and controlled pickers
        :type n_pickers: int
        :param msg_bits: Number of communication bits for each agent
        :type msg_bits: int
        :param sensor_range: Range of each agents observation
        :type sensor_range: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param layout: A string for a custom warehouse layout. X are shelve locations, dots are corridors, and g are the goal locations. Ignores shelf_columns, shelf_height and shelf_rows when used.
        :type layout: str
        :param observation_type: Specifies type of observations
        :param image_observation_layers: Specifies types of layers observed if image-observations
            are used
        :type image_observation_layers: List[ImageLayer]
        :param image_observation_directional: Specifies whether image observations should be
            rotated to be directional (agent perspective) if image-observations are used
        :type image_observation_directional: bool
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """
        self.render_mode = render_mode
        self.action_masking_level = action_masking_level
        self.sample_collection = sample_collection
        self.targets_vam = targets_vam

        self.observation_type = observation_type
        self.agents_can_clash = agents_can_clash


        self.goals: list[tuple[int, int]] = []

        self.n_agvs = n_agvs
        self.n_pickers = n_pickers
        self.n_agents = n_agvs + n_pickers

        if not layout:
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        else:
            self._make_layout_from_str(layout)
        

        if n_pickers > 0:
            self._agent_types = [AgentType.AGV for _ in range(n_agvs)] + [AgentType.PICKER for _ in range(n_pickers)]
        else:
            self._agent_types = [AgentType.AGENT for _ in range(self.n_agents)]

        self.possible_agents = [f"{at.name}_{i}" for i, at in enumerate(self._agent_types, 1)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        assert msg_bits == 0
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps: int | None = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_range = (0, 1)
        self.no_need_return_item = False
        self.fixing_clash_time = 4
        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps
        
        self.normalised_coordinates = normalised_coordinates


        self.request_queue_size = request_queue_size
        self.request_queue = []

        self.agents_list: list[Agent] = []
        
        # self.targets = np.zeros(len(self.item_loc_dict), dtype=int)
        self.stuck_count = []
        self._stuck_threshold = 5
        # default values:
        self.rack_groups = find_sections(list([loc for loc in self.item_loc_dict.values() if (loc[1], loc[0]) not in self.goals]))

        self.renderer = None

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # for performance reasons we
        # can flatten the obs vector
        if self.observation_type == "flattened":
            sa_observation_space = self.get_flattened_obs_space(agent)
        elif self.observation_type == "status":
            sa_observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        elif self.observation_type == "identifier":
            agent_type = agent.split("_")[0]
            if agent_type == "AGV":
                sa_observation_space = spaces.Box(low=0, high=1, shape=(2 + self.n_agvs,), dtype=np.float32)
            elif agent_type == "PICKER":
                sa_observation_space = spaces.Box(low=0, high=1, shape=(2 + self.n_pickers,), dtype=np.float32)
        elif self.observation_type == "none":
            sa_observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        else:
            raise NotImplementedError(f"Observation space for {self.observation_type} type not defined")

        obs_space = {"observation": sa_observation_space}
        if self.action_masking_level > 0:
            obs_space["action_mask"] = spaces.Box(0, 1, (self.action_space(agent).n,), dtype=bool)
        if self.sample_collection == "masking":
            obs_space["sample_mask"] = spaces.Box(0, 1, (1,), dtype=bool)
        return spaces.Dict(obs_space)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> spaces.Discrete:
        agent_type = agent.split("_")[0]
        if agent_type == "AGV":
            action_space = spaces.Discrete(1 + len(self.item_loc_dict))
        elif agent_type == "PICKER":
            action_space = spaces.Discrete(1 + len(self.item_loc_dict) - len(self.goals))
        return action_space
        
    def _make_layout_from_params(self, shelf_columns, shelf_rows, column_height):
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"
        self.extra_rows = 2
        self._extra_rows_columns = 1
        self.grid_size = (
            (column_height + 1) * shelf_rows + 2 + self.extra_rows,
            (2 + 1) * shelf_columns + 1,
        )
        
        self.grid_size = (
            1 + (column_height + 1 + self._extra_rows_columns) * shelf_rows + 1 + self._extra_rows_columns + self.extra_rows,
            (2 + 1 + self._extra_rows_columns) * shelf_columns + 1  + self._extra_rows_columns,
        )
        self.column_height = column_height
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        # Goals under racks
        accepted_x = []
        for i in range(0, self.grid_size[1],  3 + self._extra_rows_columns):
            accepted_x.append(i)
            for j in range(self._extra_rows_columns):
                accepted_x.append(i+j+1)

        accepted_y = []
        for i in range(0, self.grid_size[0],  1 + self._extra_rows_columns + column_height):
            accepted_y.append(i)
            for j in range(self._extra_rows_columns):
                accepted_y.append(i+j+1)

        self.goals = {
            (i, self.grid_size[0] - 1)
            for i in range(self.grid_size[1]) if not i in accepted_x
        }
        self.num_goals = len(self.goals)

        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        def highway_func(x, y):
            clauses = [
                ((x < 1 + self._extra_rows_columns or x >= self.grid_size[1] - 1 - self._extra_rows_columns) 
                or (y < 1 + self._extra_rows_columns or y >= self.grid_size[0] - 1 - self._extra_rows_columns)),
                x in accepted_x,  # vertical highways
                y in accepted_y,  # vertical highways
                (y >= self.grid_size[0] - 1 - self.extra_rows),  # delivery row
                y in [next(iter(self.goals))[1] - i - 1 for i in range(self._extra_rows_columns)],
            ]
            return any(clauses)

        # highway_func = lambda x, y: (
        #     ((x < 1 + self._extra_rows_columns or x >= self.grid_size[1] - 1 - self._extra_rows_columns) 
        #      or (y < 1 + self._extra_rows_columns or y >= self.grid_size[0] - 1 - self._extra_rows_columns))
        #     or x in accepted_x  # vertical highways
        #     or y in accepted_y  # vertical highways
        #     or (y >= self.grid_size[0] - 1 - self.extra_rows)  # delivery row
        #     or y in [self.goals[0][1] - i - 1 for i in range(self._extra_rows_columns)]
        # )
        item_loc_index = 1
        self.item_loc_dict = {}
        for y, x in self.goals:
            self.item_loc_dict[item_loc_index] = (x, y)
            item_loc_index+=1
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = highway_func(x, y)
                if not highway_func(x, y) and (x, y) not in self.goals:
                    self.item_loc_dict[item_loc_index] = (y, x)
                    item_loc_index+=1
    
    def _make_layout_from_str(self, layout):
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "gx."
                if char.lower() == "g":
                    self.goals.append((x, y))
                    self.highways[y, x] = 1
                elif char.lower() == ".":
                    self.highways[y, x] = 1

        assert len(self.goals) >= 1, "At least one goal is required"

    def get_flattened_obs_space(self, agent: str) -> spaces.Box:
        agent_type = agent.split("_")[0]
        if agent_type == "AGV":
            obs_len_agent_id = self.n_agvs
        elif agent_type == "PICKER":
            obs_len_agent_id = self.n_pickers

        # obs_len_agent_id=1

        obs_len_location = 2
        obs_len_per_agvs = 4 + 2 * obs_len_location
        obs_len_per_pickers = 2 * obs_len_location
        obs_len_per_shelf = 2

        n_shelves = len(self.item_loc_dict) - len(self.goals)

        obs_len = (
            obs_len_agent_id
            + obs_len_per_agvs * self.n_agvs
            + obs_len_per_pickers * self.n_pickers
            + obs_len_per_shelf * n_shelves
        )

        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]
    
    def get_observations(self):
        self.set_observation_requirements()
        if self.sample_collection == "relevant":
            observations = {f"{agent.type.name}_{agent.id}": self.get_agent_observation(agent) for agent in self.agents_list if not agent.busy}
        else:
            observations = {f"{agent.type.name}_{agent.id}": self.get_agent_observation(agent) for agent in self.agents_list}
        return observations

    def get_agent_observation(self, agent: Agent):
        if self.observation_type == "flattened":
            obs = self.get_flattened_obs(agent)
        elif self.observation_type == "none":
            obs = np.array([0, 0], dtype=np.float32)
        elif self.observation_type == "status":
            obs = np.array([agent.carrying_shelf is not None, agent.to_deliver], dtype=np.float32)
        elif self.observation_type == "identifier":
            status_obs = [agent.carrying_shelf is not None, agent.to_deliver]
            if agent.type == AgentType.AGV:
                id_obs = [0] * self.n_agvs
                agent_id = agent.id
            elif agent.type == AgentType.PICKER:
                id_obs = [0] * self.n_pickers
                agent_id = agent.id - self.n_agvs
            id_obs[agent_id - 1] = 1
            obs = np.array(status_obs + id_obs, dtype=np.float32)

        obs = {"observation": obs}
        if self.sample_collection == "masking":
            obs["sample_mask"] = not agent.busy
        if self.action_masking_level > 0:
            obs["action_mask"] = self.get_action_mask(agent)

        return obs

    def get_flattened_obs(self, agent: Agent):
        # write flattened observations
        obs = []

        # Agent self observation
        if agent.type == AgentType.AGV:
            id_obs = [0] * self.n_agvs
            agent_id = agent.id
            if agent.carrying_shelf is not None:
                obs.extend([1, int(agent.carrying_shelf in self.request_queue), agent.to_deliver])
            else:
                obs.extend([0, 0, agent.to_deliver])
            obs.append(agent.req_action == Action.TOGGLE_LOAD)
        elif agent.type == AgentType.PICKER:
            id_obs = [0] * self.n_pickers
            agent_id = agent.id - self.n_agvs
        id_obs[agent_id - 1] = 1
        obs.extend(id_obs)
        # obs.append(agent.id)

        obs.extend([agent.y, agent.x])
        if self.targets[agent.id - 1] != 0:
            obs.extend(self.item_loc_dict[self.targets[agent.id - 1]])
        else:
            obs.extend([0, 0])
        # Others observation
        for i in range(self.n_agents):
            agent_ = self.agents_list[i]
            if agent_.id != agent.id:
                if agent_.type == AgentType.AGV:
                    if agent_.carrying_shelf:
                        obs.extend([1, int(agent_.carrying_shelf in self.request_queue), agent_.to_deliver])
                    else:
                        obs.extend([0, 0, agent_.to_deliver])
                    obs.append(agent_.req_action == Action.TOGGLE_LOAD)
                obs.extend([agent_.y, agent_.x])
                if self.targets[agent_.id - 1] != 0:
                    obs.extend(self.item_loc_dict[self.targets[agent_.id - 1]])
                else:
                    obs.extend([0, 0])
        # Shelves observation
        for group in self.rack_groups:
            for (x, y) in group:
                id_shelf = self.grid[_LAYER_SHELFS, x, y]
                if id_shelf == 0:
                    obs.extend([0, 0])
                else:
                    obs.extend(
                        [1.0 , int(self.shelfs[id_shelf - 1] in self.request_queue)]
                    )
                # id_carried_shelf = self.grid[_LAYER_CARRIED_SHELFS, x, y]
                # if id_carried_shelf == 0:
                #     obs.skip(1)
                # else:
                #     obs.write([1.0])
        return np.array(obs, dtype=np.float32)
    
    def find_path(self, start, goal, agent, care_for_agents = True):
        grid = np.zeros(self.grid_size)
        # if agent.type in [AgentType.AGV, AgentType.AGENT]:
        if care_for_agents:
            grid += self.grid[_LAYER_AGENTS]
            grid += self.grid[_LAYER_PICKERS]
            if agent.type == AgentType.PICKER:
                grid[goal[0], goal[1]] -= self.grid[_LAYER_AGENTS, goal[0], goal[1]]
            else:
                grid[goal[0], goal[1]] -= self.grid[_LAYER_PICKERS, goal[0], goal[1]]

        special_case_jump = False
        if agent.type == AgentType.PICKER:
            # Agents can only travel through highways if carrying a shelf
            for x in range(self.grid_size[1]):
                for y in range(self.grid_size[0]):
                    grid[y, x] += not self._is_highway(x, y)
            grid[goal[0], goal[1]] -= not self._is_highway(goal[1], goal[0])
            if   agent.type == AgentType.PICKER and  ((not self._is_highway(start[1], start[0])) and goal[0] == start[0] and abs(goal[1] - start[1]) == 1): # Ban "jumps" from on shelf to another
                special_case_jump = True
            for i in range(self.grid_size[1]):
                grid[self.grid_size[0] - 1, i] = 1

        grid[start[0], start[1]] = 0
        if not special_case_jump:
            grid = [list(map(int, l)) for l in (grid!=0)]
            grid = np.array(grid, dtype=np.float32)
            grid[np.where(grid == 1)] = np.inf
            grid[np.where(grid == 0)] = 1
            astar_path = pyastar2d.astar_path(grid, start, goal, allow_diagonal=False) # returns None if cant find path
            if astar_path is not None:
                astar_path = [tuple(x) for x in list(astar_path)] # convert back to other format
                astar_path = astar_path[1:]
        else:
            special_start = None
            if self._is_highway(start[1] - 1, start[0]):
                special_start = (start[0], start[1] - 1)
            if self._is_highway(start[1] + 1, start[0]):
                special_start = (start[0], start[1] + 1)
            grid[start[0], start[1]] = 1
            grid[special_start[0], special_start[1]] = 0
            grid = [list(map(int, l)) for l in (grid!=0)]
            grid = np.array(grid, dtype=np.float32)
            grid[np.where(grid == 1)] = np.inf
            grid[np.where(grid == 0)] = 1
            astar_path = pyastar2d.astar_path(grid, special_start, goal, allow_diagonal=False)
            if astar_path is not None:
                astar_path = [tuple(x) for x in list(astar_path)] # convert back to other format
            astar_path = astar_path
        if astar_path:
            return [(x, y) for y, x in astar_path]
        else:
            return []

    def _recalc_grid(self):
        self.grid[:] = 0
        carried_shelf_ids = [agent.carrying_shelf.id for agent in self.agents_list if agent.carrying_shelf]
        for s in self.shelfs:
            if s.id not in carried_shelf_ids:
                self.grid[_LAYER_SHELFS, s.y, s.x] = s.id
        for agent in self.agents_list:
            if agent.type == AgentType.PICKER:
                self.grid[_LAYER_PICKERS, agent.y, agent.x] = agent.id
            else:
                self.grid[_LAYER_AGENTS, agent.y, agent.x] = agent.id
            if agent.carrying_shelf:
                self.grid[_LAYER_CARRIED_SHELFS, agent.y, agent.x] = agent.carrying_shelf.id

    def set_observation_requirements(self):
        empty_item_map = [0] * (len(self.item_loc_dict) - len(self.goals))
        request_item_map = [0] * (len(self.item_loc_dict) - len(self.goals))
        requested_shelf_ids = {shelf.id for shelf in self.request_queue}

        for id_, coords in self.item_loc_dict.items():
            if (coords[1], coords[0]) not in self.goals:
                if self.grid[_LAYER_SHELFS, coords[0], coords[1]] in requested_shelf_ids:
                    request_item_map[id_ - len(self.goals) - 1] = 1
                if self.grid[_LAYER_SHELFS, coords[0], coords[1]] == 0 and (self.grid[_LAYER_CARRIED_SHELFS, coords[0], coords[1]] == 0 or self.agents_list[self.grid[_LAYER_AGENTS, coords[0], coords[1]] - 1].req_action not in [Action.NOOP, Action.TOGGLE_LOAD]):
                    empty_item_map[id_ - len(self.goals) - 1] = 1

        self.requested_items_list = request_item_map
        self.empty_items_list = empty_item_map

        targets_list = [tar - len(self.goals) - 1 for tar in self.targets]
        self.agv_targets = set(targets_list[:self.n_agvs])
        self.picker_targets = set(targets_list[self.n_agvs:])

    
    def get_action_mask(self, agent: Agent):
        if agent.type == AgentType.AGV:
            return self.get_agv_action_mask(agent)
        if agent.type == AgentType.PICKER:
            return self.get_picker_action_mask()
        raise ValueError(f"Action mask not implemented for agent type: {agent.type}")
    
    def get_picker_action_mask(self):
        noop_mask = [True]

        if self.action_masking_level > 1:
            item_mask = [(i in self.agv_targets) and (i not in self.picker_targets) for i in range(len(self.item_loc_dict) - len(self.goals))]
        elif self.action_masking_level == 1:
            item_mask = [(i in self.agv_targets) for i in range(len(self.item_loc_dict) - len(self.goals))]

        return np.array(noop_mask + item_mask, dtype=bool)

    def get_agv_action_mask(self, agent: Agent):
        agent_is_carrying = (agent.carrying_shelf is not None)
        agent_to_return = agent_is_carrying and not agent.to_deliver

        noop_mask = [True]

        if self.action_masking_level == 3:
            goal_mask = [agent.to_deliver] * len(self.goals)
            if agent.to_deliver:
                item_mask = [False] * len(self.requested_items_list)
            elif agent_to_return:
                item_mask = [empty and (i not in self.agv_targets) for i, empty in enumerate(self.empty_items_list)]
            else:
                item_mask = [requested and (i not in self.agv_targets) for i, requested in enumerate(self.requested_items_list)]
        elif self.action_masking_level == 2:
            goal_mask = [agent_is_carrying] * len(self.goals)
            item_mask = [(empty if agent_is_carrying else requested) and (i not in self.agv_targets) for i, (empty, requested) in enumerate(zip(self.empty_items_list, self.requested_items_list))]
        elif self.action_masking_level == 1:
            goal_mask = [agent_is_carrying] * len(self.goals)
            item_mask = [(empty if agent_is_carrying else requested) for empty, requested in zip(self.empty_items_list, self.requested_items_list)]


        return np.array(noop_mask + goal_mask + item_mask, dtype=bool)

    def reset(self, seed=None, options=None):
        if not hasattr(self, "np_random"):
            self.np_random = default_rng(seed) if seed else default_rng(0)
        self.agents = self.possible_agents

        # self.episodic_return = Counter({agent_id: 0 for agent_id in self.agent_name_mapping.keys()})
        # self.aggregated_reward = {agent_id: 0 for agent_id in self.agent_name_mapping.keys()}
        # self.episode_stats = defaultdict(0)
        self.episodic_return = Counter()
        self.aggregated_reward = Counter()
        self.episode_stats = Counter()

        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9

        # make the shelfs
        self.shelfs = [
            Shelf(x, y)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y)
        ]
        self._higway_locs = np.array([(y, x) for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            ) if self._is_highway(x, y)])
        
        # Spawn agents on higwahy locations 
        agent_loc_ids = self.np_random.choice(
            np.arange(len(self._higway_locs)),
            size=self.n_agents,
            replace=False,
        )
        agent_locs = [self._higway_locs[agent_loc_ids, 0], self._higway_locs[agent_loc_ids, 1]]
        # and direction
        agent_dirs = self.np_random.choice([d for d in Direction], size=self.n_agents)
        self.agents_list = [
            Agent(x, y, dir_, self.msg_bits, agent_type = agent_type)
            for y, x, dir_, agent_type in zip(*agent_locs, agent_dirs, self._agent_types)
        ]
        self._recalc_grid()

        self.request_queue = list(
            self.np_random.choice(self.shelfs, size=self.request_queue_size, replace=False)
        )
        self.targets = np.zeros(len(self.agents_list), dtype=int)
        self.stuck_count = [[0, (agent.x, agent.y)] for agent in self.agents_list]

        observations = self.get_observations()
        infos = {f"{agent.type.name}_{agent.id}": {"busy": agent.busy} for agent in self.agents_list}
        return observations, infos

    def resolve_move_conflict(self, agent_list):
        # # stationary agents will certainly stay where they are
        # stationary_agents = [agent for agent in self.agents_list if agent.action != Action.FORWARD]

        # forward agents will move only if they avoid collisions
        # forward_agents = [agent for agent in self.agents_list if agent.action == Action.FORWARD]
        
        commited_agents = set()

        G = nx.DiGraph()
        
        for agent in agent_list:
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)
            G.add_edge(start, target)

        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        for comp in wcomps:
            try:
                # if we find a cycle in this component we have to
                # commit all nodes in that cycle, and nothing else
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # we have a situation like this: [A] <-> [B]
                    # which is physically impossible. so skip
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[_LAYER_AGENTS, start_node[1], start_node[0]]
                    # action = self.agents_list[agent_id - 1].req_action
                    # print(f"{agent_id}: C {cycle} {action}")
                    if agent_id > 0:
                        commited_agents.add(agent_id)
                        continue
                    picker_id = self.grid[_LAYER_PICKERS, start_node[1], start_node[0]]
                    if picker_id > 0:
                        commited_agents.add(picker_id)
                        continue
            except nx.NetworkXNoCycle:

                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)
                        continue
                    picker_id = self.grid[_LAYER_PICKERS, y, x]
                    if picker_id:
                        commited_agents.add(picker_id)
        clashes = 0
        for agent in agent_list:
            for other in agent_list:
                if agent.id != other.id:
                    agent_new_x, agent_new_y = agent.req_location(self.grid_size)
                    other_new_x, other_new_y = other.req_location(self.grid_size)
                    # Clash fixing logic
                    if agent.path and ((agent_new_x, agent_new_y) in [(other.x, other.y), (other_new_x, other_new_y)]): 
                        # If we are in a rack and one of the agents is a picker we ignore clashses, assumed behaviour is Picker is loading
                        if not self._is_highway(agent_new_x, agent_new_y) and (agent.type == AgentType.PICKER or other.type == AgentType.PICKER) and agent.type != other.type:
                            # Allow Pickers to step over AGVs (if no other Picker at that shelf location) or AGVs to step over Pickers (if no other AGV at that shelf location)
                            if ((agent.type == AgentType.PICKER and self.grid[_LAYER_PICKERS, agent_new_y, agent_new_x] in [0, agent.id]) 
                                or (agent.type == AgentType.AGV and self.grid[_LAYER_AGENTS, agent_new_y, agent_new_x] in [0, agent.id])):
                                commited_agents.add(agent.id)
                                continue
                        # If the agent's next action bumps it into another agent
                        if (agent_new_x, agent_new_y) == (other.x, other.y):
                            agent.req_action = Action.NOOP # Stop the action
                            # Check if the clash is not solved naturaly by the other agent moving away
                            if (other_new_x, other_new_y) in [(agent.x, agent.y), (agent_new_x, agent_new_y)] and not other.req_action in (Action.LEFT, Action.RIGHT):
                                if other.fixing_clash == 0:# If the others are not already fixing the clash
                                    clashes+=1
                                    agent.fixing_clash = self.fixing_clash_time # Agent start time for clash fixing
                                    new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), agent)
                                    if new_path != []: # If the agent can find an alternative path, assign it if not let the other solve the clash
                                        agent.path = new_path
                                    else:
                                        agent.fixing_clash = 0
                        elif (agent_new_x, agent_new_y) == (other_new_x, other_new_y) and (agent_new_x, agent_new_y) != (agent.x, agent.y): 
                            # If the agent's next action bumps it into another agent position after they take actions simultaneously
                            if agent.fixing_clash == 0 and other.fixing_clash == 0:
                                agent.req_action = Action.NOOP # If the agent's actions leads them in the position of another STOP
                                agent.fixing_clash = self.fixing_clash_time  # Agent wait one step while the other moves into place

        commited_agents = set([self.agents_list[id_ - 1] for id_ in commited_agents])
        failed_agents = set(agent_list) - commited_agents
        for agent in failed_agents:
            agent.req_action = Action.NOOP
        return clashes
    def step(
        self, macro_actions: dict[str, Action]
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        # Logic for Macro Actions
        selected_actions = defaultdict(set)
        duplicate_action_count = defaultdict(int)
        for agent_id in self.agents:
            #NOTE: hacky way to select noop if action not present, needs something better
            macro_action = macro_actions.get(agent_id, 0)
            agent = self.agents_list[self.agent_name_mapping[agent_id]]
            # Initialize action for step
            agent.req_action = Action.NOOP
            # Collision avoidance logic
            if agent.fixing_clash > 0:
                agent.fixing_clash -= 1
                # continue
            if not agent.busy:
                if macro_action != 0:
                    if agent.type == AgentType.PICKER:
                        macro_action = macro_action + len(self.goals)

                    if macro_action in selected_actions[agent.type]:
                        duplicate_action_count[agent.type] += 1
                    selected_actions[agent.type].add(macro_action)

                    agent.path = self.find_path((agent.y, agent.x), self.item_loc_dict[macro_action], agent, care_for_agents=False)
                    # If not path was found refuse location
                    if agent.path == []:
                        agent.busy = False
                    else:
                        agent.busy = True
                        agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                        self.targets[agent.id-1] = macro_action
                        self.stuck_count[agent.id - 1] = [0, (agent.x, agent.y)]
            else:
                # Check agent finished the give path if not continue the path
                if agent.path == []:
                    #self.targets[agent.id-1] = 0
                    if agent.type != AgentType.PICKER:
                        agent.req_action = Action.TOGGLE_LOAD
                    if agent.type != AgentType.AGV:
                        agent.busy = False
                else:
                    agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                    # If agent is at the end of a path and carrying a shelf and the target location is already occupied restart agent
                if agent.path and len(agent.path) == 1:
                    if agent.carrying_shelf and self.grid[_LAYER_SHELFS, agent.path[-1][1], agent.path[-1][0]]:
                        agent.req_action = Action.NOOP
                        agent.busy = False
                    if agent.type == AgentType.PICKER:
                        if (self.grid[_LAYER_AGENTS, agent.path[-1][1], agent.path[-1][0]] == 0 or self.agents_list[self.grid[_LAYER_AGENTS, agent.path[-1][1], agent.path[-1][0]] - 1].req_action != Action.TOGGLE_LOAD):
                            agent.req_action = Action.NOOP
                            # agent.busy = False
                        elif self.grid[_LAYER_AGENTS, agent.path[-1][1], agent.path[-1][0]] != 0 and self.agents_list[self.grid[_LAYER_AGENTS, agent.path[-1][1], agent.path[-1][0]] - 1].req_action == Action.TOGGLE_LOAD:
                            self.stuck_count[agent.id - 1] = [0, (agent.x, agent.y)]

        #  agents that can_carry should not collide
        if self.agents_can_clash:
            clashes_count = self.resolve_move_conflict(self.agents_list)
        else:
            clashes_count = 0

        # Restart agents if they are stuck at the same position
        # This can happen when their goal is occupied after reaching their last step/re-calculating a path
        stucks_count = 0
        agvs_distance_travelled = 0
        pickers_distance_travelled = 0
        for agent in self.agents_list:
            if agent.busy: # Don't count path calculation/fixing steps
                if agent.req_action not in (Action.LEFT, Action.RIGHT): # Don't count changing directions
                    if agent.req_action!=Action.TOGGLE_LOAD or (agent.x, agent.y) in self.goals: # Don't count loading or changing directions
                        pos = self.stuck_count[agent.id - 1][1]
                        if agent.x == pos[0] and agent.y == pos[1]:
                            self.stuck_count[agent.id - 1][0] += 1
                        else:
                            self.stuck_count[agent.id - 1] = [0, (agent.x, agent.y)]
                            if agent.type == AgentType.PICKER:
                                pickers_distance_travelled += 1
                            else:
                                agvs_distance_travelled += 1
                        if self.stuck_count[agent.id - 1][0] > self._stuck_threshold and self.stuck_count[agent.id - 1][0] < self._stuck_threshold + self.column_height + 2: # Time to get out of aisle 
                            agent.req_action = Action.NOOP
                            if agent.path:
                                new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), agent)
                                if new_path:
                                    agent.path = new_path
                                    if agent.type == AgentType.PICKER and len(agent.path) == 1:
                                        continue
                                    self.stuck_count[agent.id - 1] = [0, (agent.x, agent.y)]
                                    continue
                            else:
                                stucks_count += 1
                                agent.busy = False
                                self.stuck_count[agent.id - 1] = [0, (agent.x, agent.y)]

                        if self.stuck_count[agent.id - 1][0] > self._stuck_threshold + self.column_height + 2: # Time to get out of aisle 
                            stucks_count += 1
                            self.stuck_count[agent.id - 1] = [0, (agent.x, agent.y)]
                            agent.req_action = Action.NOOP
                            agent.busy = False

        rewards = np.zeros(self.n_agents, dtype=np.float32)
        # Add step penalty
        rewards -= 0.001
        for agent in self.agents_list:
            agent.prev_x, agent.prev_y = agent.x, agent.y
            if agent.req_action == Action.FORWARD:
                agent.x, agent.y = agent.req_location(self.grid_size)
                agent.path = agent.path[1:]
                if agent.carrying_shelf:
                    agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf and agent.type != AgentType.PICKER:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                if shelf_id:
                    if agent.type == AgentType.AGV:
                        picker_id = self.grid[_LAYER_PICKERS, agent.y, agent.x]
                        if picker_id:
                            agent.carrying_shelf = self.shelfs[shelf_id - 1]
                            self.grid[_LAYER_SHELFS, agent.y, agent.x] = 0
                            self.grid[_LAYER_CARRIED_SHELFS, agent.y, agent.x] = shelf_id
                            agent.busy = False
                            agent.to_deliver = True
                            # Reward Pickers for loading shelf
                            if self.reward_type == RewardType.GLOBAL:
                                rewards += 0.5
                            elif self.reward_type == RewardType.INDIVIDUAL:
                                rewards[picker_id - 1] += 0.1
                                # rewards[agent.id - 1] += 0.5 #NOTE: added by Bram to provide reward for AGVs that pickup a shelf
                    elif agent.type == AgentType.AGENT:
                        agent.carrying_shelf = self.shelfs[shelf_id - 1]
                        agent.busy = False
                else:
                    agent.busy = False
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf and agent.type != AgentType.PICKER:
                picker_id = self.grid[_LAYER_PICKERS, agent.y, agent.x]
                if (agent.x, agent.y) in self.goals:
                    agent.busy = False
                    agent.to_deliver = False
                    continue
                if self.grid[_LAYER_SHELFS, agent.y, agent.x] != 0:
                    agent.busy = False
                    continue
                if not self._is_highway(agent.x, agent.y):
                    if agent.type == AgentType.AGENT:
                        self.grid[_LAYER_SHELFS, agent.y, agent.x] =  agent.carrying_shelf.id
                        self.grid[_LAYER_CARRIED_SHELFS, agent.y, agent.x] = 0
                        agent.carrying_shelf = None
                        agent.busy = False
                    if agent.type == AgentType.AGV and picker_id:
                        self.grid[_LAYER_SHELFS, agent.y, agent.x] =  agent.carrying_shelf.id
                        self.grid[_LAYER_CARRIED_SHELFS, agent.y, agent.x] = 0
                        agent.carrying_shelf = None
                        agent.busy = False
                        # Reward Pickers for un-loading shelf
                        if self.reward_type == RewardType.GLOBAL:
                            rewards += 0.5
                        elif self.reward_type == RewardType.INDIVIDUAL:
                            rewards[picker_id - 1] += 0.1
                    if agent.has_delivered and self.reward_type == RewardType.TWO_STAGE:
                        # rewards[agent.id - 1] += 0.5
                        raise NotImplementedError('TWO_STAGE reward not implemenred for diverse rware')
                    agent.has_delivered = False
        shelf_delivered = False
        shelf_deliveries = 0
        for y, x in self.goals:
            shelf_id = self.grid[_LAYER_CARRIED_SHELFS, x, y]
            if not shelf_id:
                continue
            shelf = self.shelfs[shelf_id - 1]

            if shelf not in self.request_queue:
                continue
            # a shelf was successfully delived.
            shelf_delivered = True
            shelf_deliveries += 1
            # remove from queue and replace it
            carried_shels = [agent.carrying_shelf for agent in self.agents_list if agent.carrying_shelf]
            new_shelf_candidates = list(set(self.shelfs) - set(self.request_queue) - set(carried_shels)) # sort so self.np_random with seed is repeatable
            new_shelf_candidates.sort(key = lambda x: x.id)
            new_request = self.np_random.choice(new_shelf_candidates)
            self.request_queue[self.request_queue.index(shelf)] = new_request

            if self.no_need_return_item:
                agent.carrying_shelf = None
                for sx, sy in zip(
                    np.indices(self.grid_size)[0].reshape(-1),
                    np.indices(self.grid_size)[1].reshape(-1),
                ): 
                    if not self._is_highway(sy, sx) and not self.grid[_LAYER_SHELFS, sy, sx]:
                        print(f"{sx}-{sy}")
                        self.shelfs[shelf_id - 1].x = sx
                        self.shelfs[shelf_id - 1].y = sy
                        self.grid[_LAYER_SHELFS, sy, sx] = shelf_id
                        break
            # also reward the agents
            if self.reward_type == RewardType.GLOBAL:
                rewards += 1
            elif self.reward_type == RewardType.INDIVIDUAL:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                rewards[agent_id - 1] += 1
            elif self.reward_type == RewardType.TWO_STAGE:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                self.agents_list[agent_id - 1].has_delivered = True
                rewards[agent_id - 1] += 1
        self._recalc_grid()

        if shelf_delivered:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        self._cur_steps += 1

        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            episode_done = True
        else:
            episode_done = False

        
        #NOTE: Why count TOGGLE_LOAD?
        agvs_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents_list[:self.n_agvs]])
        pickers_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents_list[self.n_agvs:]])

        # sum counters
        step_rewards = {agent_id: rewards[i] for agent_id, i in self.agent_name_mapping.items()}
        self.aggregated_reward.update(step_rewards)
        self.episodic_return.update(step_rewards)
        self.episode_stats.update({
            "shelf_deliveries": shelf_deliveries,
            "clashes": clashes_count,
            "stucks": stucks_count,
            "agvs_distance_travelled": agvs_distance_travelled,
            "pickers_distance_travelled": pickers_distance_travelled,
            "agvs_idle_time": agvs_idle_time,
            "pickers_idle_time": pickers_idle_time,
            "agvs_duplicate_shelf_actions": duplicate_action_count[AgentType.AGV],
            "pickers_duplicate_shelf_actions": duplicate_action_count[AgentType.PICKER],
        })
    
        # Construct return values
        observations = self.get_observations()

        if self.sample_collection == "relevant":
            returning_agents = list(observations.keys())
            rewards = {agent_id: self.aggregated_reward[agent_id] for agent_id in returning_agents}
            for agent_id in returning_agents:
                self.aggregated_reward[agent_id] = np.float32(0)
        else:
            returning_agents = self.agents
            rewards = step_rewards

        terminateds = {agent_id: episode_done for agent_id in returning_agents}
        truncateds = {agent_id: episode_done for agent_id in returning_agents}
        infos = {f"{agent.type.name}_{agent.id}": {"busy": agent.busy} for agent in self.agents_list}
        infos["__step_common__"] = {
            "shelf_deliveries": shelf_deliveries,
            "clashes": clashes_count,
            "stucks": stucks_count,
            "agvs_distance_travelled": agvs_distance_travelled,
            "pickers_distance_travelled": pickers_distance_travelled,
            "agvs_idle_time": agvs_idle_time,
            "pickers_idle_time": pickers_idle_time
        }

        if episode_done:
            for agent_id in self.agents:
                infos[agent_id]["episode"] = {"return": self.episodic_return[agent_id], "length": self._cur_steps}
            infos["__common__"] = self.episode_stats
            pickrate = self.episode_stats["shelf_deliveries"] * 3600 / (5 * self._cur_steps)
            infos["__common__"]["pickrate"] = pickrate
            terminateds = truncateds = {agent_id: True for agent_id in self.agents}
        else:
            terminateds = truncateds = {agent_id: False for agent_id in returning_agents}

        return observations, rewards, terminateds, truncateds, infos

    def render(self):
        if not self.renderer:
            from tarware.rendering import Viewer

            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=(self.render_mode=="rgb_array"))

    def close(self):
        if self.renderer:
            self.renderer.close()


class RepeatedWarehouse(Warehouse):
    def step(self, macro_actions: dict[str, Action]):

        observations, rewards, terms, truncs, infos = super().step(macro_actions)

        env_done = all([terms.get(agent_id, False) for agent_id in self.agents])

        if env_done:
            observations, reset_infs = self.reset()
            infos = {agent_id: reset_infs.get(agent_id, {}) | info for agent_id, info in infos.items()}

        return observations, rewards, terms, truncs, infos
