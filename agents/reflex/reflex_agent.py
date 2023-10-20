import numpy as np
from queue import PriorityQueue
import random
from typing import Dict, Any, Tuple, List
from gym import spaces

from pommerman import agents
from pommerman.constants import Item, Action

import agents.reflex.util as util
from agents.reflex.util import FindItemPredicate, FindWoodPredicate
from agents.verbose_agent import VerboseAgent

class FindEnemyPredicate():
    def __init__(self, postition: List[int]) -> None:
        self.position = postition

    def test(self, board: np.ndarray, position: Tuple[int, int]) -> bool:
        return abs(self.position[0] - position[0]) <  3 and abs(self.position[1] - position[1]) <  3



class ReflexAgent(VerboseAgent):
    """
    This is the class of your agent. During the tournament an object of this class
    will be created for every game your agents plays.
    If you exceed 500 MB of main memory used, your agent will crash.

    Args:
        ignore the arguments passed to the constructor
        in the constructor you can do initialisation that must be done before a game starts
    """
    def __init__(self, *args, **kwargs):
        super(ReflexAgent, self).__init__(*args, **kwargs)


    def act(self, obs, action_space):
        """
        Every time your agent is required to send a move, this method will be called.
        You have 0.5 seconds to return a move, otherwise no move will be played.

        Parameters
        ----------
        obs: dict
            keys:
                'alive': {list:2}, board ids of agents alive
                'board': {ndarray: (11, 11)}, board representation
                'bomb_blast_strength': {ndarray: (11, 11)}, describes range of bombs
                'bomb_life': {ndarray: (11, 11)}, shows ticks until bomb explodes
                'bomb_moving_direction': {ndarray: (11, 11)}, describes moving direction if bomb has been kicked
                'flame_life': {ndarray: (11, 11)}, ticks until flame disappears
                'game_type': {int}, irrelevant for you, we only play FFA version
                'game_env': {str}, irrelevant for you, we only use v0 env
                'position': {tuple: 2}, position of the agent (row, col)
                'blast_strength': {int}, range of own bombs         --|
                'can_kick': {bool}, ability to kick bombs             | -> can be improved by collecting items
                'ammo': {int}, amount of bombs that can be placed   --|
                'teammate': {Item}, irrelevant for you
                'enemies': {list:3}, possible ids of enemies, you only have one enemy in a game!
                'step_count': {int}, if 800 steps were played then game ends in a draw (no points)

        action_space: spaces.Discrete(6)
            action_space.sample() returns a random move (int)
            6 possible actions in pommerman (integers 0-5)

        Returns
        -------
        action: int
            Stop (0): This action is a pass.
            Up (1): Move up on the board.
            Down (2): Move down on the board.
            Left (3): Move left on the board.
            Right (4): Move right on the board.
            Bomb (5): Lay a bomb.
        """

        # first we need to reformat the observations space for further processing
        my_position: Tuple[int, int] = tuple(obs['position'])
        board: np.ndarray = np.array(obs['board'])
        bomb_blast_strength: np.ndarray = np.array(obs['bomb_blast_strength'])
        bomb_life: np.ndarray = np.array(obs['bomb_life'])
        enemy: Item = obs['enemies'][0]  # we only have to deal with 1 enemy
        epos: np.ndarray = np.where(obs['board'] == enemy.value)
        enemy_position: Tuple[int, int] = (epos[0][0], epos[1][0])
        ammo: int = int(obs['ammo'])
        blast_strength: int = int(obs['blast_strength'])
        steps: int = int(obs['step_count'])
        if self.log: self.log_state(obs)

        actions = self.legal_moves(my_position, board, bomb_blast_strength[my_position[0], my_position[1]] != 0, ammo)

        # we have extracted all legal moves, we now want to filter those moves,
        # that bring our agent in dangerous situations
        bombs = self._convert_bombs(bomb_life, bomb_blast_strength)
        danger_map = self._get_danger_map(board, bombs, bomb_blast_strength)
        actions = self._prune_dangerous_actions(board, bomb_blast_strength, my_position, enemy_position, actions, danger_map, blast_strength)

        # if <= 1 action is left then we can return here
        if len(actions) == 1:
            return actions[0]
        elif len(actions) == 0:
            # TODO: all moves pruned, maybe you will need some emergency handling here
            return Action.Stop.value

        
        tiles_in_range = util.get_in_range(board, my_position, blast_strength)

        def collecting_strategy(path_tolerance = 3):
            # explore and collect heuristic
            # TODO: you might find more efficient and accurate algorithms here
            # check via BFS if we can pick up an item
            goal_node = util.bfs(board, my_position, actions,
                                    FindItemPredicate([Item.Kick.value, Item.ExtraBomb.value, Item.IncrRange.value]))
            if goal_node:
                action, path_length = util.PositionNode.get_path_length_and_action(goal_node)
                goal_r, goal_c = goal_node.position
                if path_length < 4 * path_tolerance and danger_map[goal_r, goal_c] != path_length:
                    return action

            if Item.Wood.value in tiles_in_range and Action.Bomb.value in actions:
                return Action.Bomb.value

            # try to approach a wooden tile
            goal_node = util.bfs(board, my_position, actions,
                                    FindWoodPredicate(blast_strength, bomb_blast_strength))
            if goal_node:
                action, path_length = util.PositionNode.get_path_length_and_action(goal_node)
                goal_r, goal_c = goal_node.position
                if path_length < 5 * path_tolerance and danger_map[goal_r, goal_c] != path_length:
                    return action
                
        def aggresive_strategy(path_tolerance = 3.5):
            # attack heuristic
            # lay bomb if it is not a pruned action and it can hit the enemy
            #enemy_dist = util.manhattan_distance(my_position, enemy_position)
            if enemy.value in tiles_in_range and Action.Bomb.value in actions:
                return Action.Bomb.value
            # Otherwise try to get closer to the enemy
            goal_node = util.bfs(board, my_position, actions,FindEnemyPredicate(enemy_position))
            if goal_node:
                action, path_length = util.PositionNode.get_path_length_and_action(goal_node)
                if path_length < 2 * path_tolerance:
                    return action

        proposal = aggresive_strategy(2)
        if proposal: return proposal
        if steps < 250: # Agent should hurry as the game progresses
            proposal = collecting_strategy(path_tolerance=3)
            if proposal: return proposal
        else:
            proposal = aggresive_strategy(4)
            if proposal: return proposal
            else: 
                proposal = collecting_strategy(15)
                if proposal: return proposal

        # final strategy if all others fail - randomly choosing move
        proposal = aggresive_strategy(20)
        if proposal: return proposal
        proposal = collecting_strategy(35)
        if proposal: return proposal
        #print(" What should I do?")
        if Action.Bomb.value in actions: actions.remove(Action.Bomb.value)
        return random.choice(actions)
    
    def legal_moves(self, position: Tuple[int, int], board: np.ndarray, on_bomb: bool, ammo: int) -> List[int]:
        """
        Filters actions like bumping into a wall (which is equal to "Stop" action) or trying
        to lay a bomb, although there is no ammo available
        """
        all_actions = [Action.Stop.value]  # always possible
        if not on_bomb and ammo > 0:
            all_actions.append(Action.Bomb.value)

        up = position[0] - 1
        down = position[0] + 1
        left = position[1] - 1
        right = position[1] + 1

        if up >= 0 and self.is_accessible(board[up, position[1]]):
            all_actions.append(Action.Up.value)
        if down < len(board) and self.is_accessible(board[down, position[1]]):
            all_actions.append(Action.Down.value)
        if left >= 0 and self.is_accessible(board[position[0], left]):
            all_actions.append(Action.Left.value)
        if right < len(board) and self.is_accessible(board[position[0], right]):
            all_actions.append(Action.Right.value)

        return all_actions

    @staticmethod
    def is_accessible(pos_val: int) -> bool:
        return pos_val in util.ACCESSIBLE_TILES

    @staticmethod
    def _convert_bombs(bomb_life: np.ndarray, bomb_blast_strength: np.ndarray) -> PriorityQueue:
        """Convert bomb matrices in bomb queue sorted by bomb life
            sorting bombs makes calculating the danger map faster afterwards
        """
        bombs = PriorityQueue()
        locations = np.where(bomb_blast_strength > 0)
        for r, c in zip(locations[0], locations[1]):
            bombs.put((int(bomb_life[r, c]), int(bomb_blast_strength[r, c]), (r, c)))
        return bombs

    def check_position(board: np.ndarray, position: Tuple[int, int], danger_map: np.ndarray, blast_map: np.ndarray, look_ahead: int = 1) -> bool:
        r,c = position
        lim_r, lim_c = board.shape
        if np.amax(danger_map) < look_ahead: return True
        elif danger_map[r,c] == look_ahead:return False
        else: 
            next_legal_positions = [position, (min([lim_r - 1, r + 1]),c), (r,min([lim_c - 1, c + 1])), (r,max([0, c - 1])), (max([0, r - 1]), c)]
            next_legal_positions = [nlc for nlc in next_legal_positions if (board[nlc[0],nlc[1]] in util.ACCESSIBLE_TILES or board[nlc[0],nlc[1]] == Item.Wood.value and look_ahead > danger_map[r,c])\
                and not (blast_map[nlc[0],nlc[1]] and look_ahead <= danger_map[r,c])]
            for next_pos in next_legal_positions:
                if ReflexAgent.check_position(board, next_pos, danger_map, blast_map, look_ahead + 1): return True
    
    
    @staticmethod
    def _prune_dangerous_actions(board: np.ndarray, bomb_blast_strength: np.ndarray, position: Tuple[int, int], enemy_position: Tuple[int, int],
                                 actions: List[int], danger_map: np.ndarray, blast_strength: int) -> List[int]:
        # TODO: finding good pruning rules here is extremely important,
        #  the version currently used is only a very rough approximation
        pruned_actions = []
        for action in actions:
            # we already know that r and c are in bounds here
            action_danger_map = np.copy(danger_map)
            action_blast_map = np.copy(bomb_blast_strength)
            bombs_to_check = [enemy_position]
            if action == Action.Bomb.value:bombs_to_check.append(position)
            for pos_to_check in bombs_to_check:    
                r_o,c_o = pos_to_check
                bomb_life = min([10, action_danger_map[r_o,c_o]])
                action_danger_map[r_o,c_o] = bomb_life
                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for dist in range(1, blast_strength):
                        r = r_o + row * dist
                        c = c_o + col * dist
                        if r < 0 or r >= len(board) or c < 0 or c >= len(board):
                            # out of border
                            break
                        elif board[r, c] in util.SOLID_TILES:
                            # solid tile stops bomb
                            action_danger_map[r, c] = min(action_danger_map[r, c] if action_danger_map[r, c] else 150, bomb_life)
                            break
                        else:
                            # update danger map
                            action_danger_map[r, c] = min(action_danger_map[r, c] if action_danger_map[r, c] else 150, bomb_life)
                action_blast_map[r_o,c_o] = blast_strength

            next_position = util.next_position(position, action)

            if ReflexAgent.check_position(board, next_position, action_danger_map, action_blast_map, 1): pruned_actions.append(action)

            """ # the following pruning rules are only heuristics (not exact rules)
            # always move away from a bomb
            if action == Action.Stop.value and bomb_blast_strength[r, c] != 0.0:
                continue
            if danger_map[r, c] <= 2:
                # too dangerous if bomb explosion happens in 2 steps
                continue
            elif action == Action.Bomb.value:
                # it is too dangerous to trigger chain reactions - avoid them
                if danger_map[r, c] < util.MAX_BOMB_LIFE:
                    continue

            # check if agent is locked in
            down_cond = r + 1 >= len(board) or \
                board[r + 1, c] in util.SOLID_TILES or \
                bomb_blast_strength[r + 1, c] != 0
            up_cond = r - 1 < 0 or \
                board[r - 1, c] in util.SOLID_TILES or \
                bomb_blast_strength[r - 1, c] != 0
            right_cond = c + 1 >= len(board) or \
                board[r, c + 1] in util.SOLID_TILES or \
                bomb_blast_strength[r, c + 1] != 0
            left_cond = c - 1 < 0 or \
                board[r, c - 1] in util.SOLID_TILES or \
                bomb_blast_strength[r, c - 1] != 0

            if not(down_cond and up_cond and right_cond and left_cond):
                pruned_actions.append(action)"""
        return pruned_actions

    @staticmethod
    def _get_danger_map(board: np.ndarray, bombs: PriorityQueue, bomb_blast_strength: np.ndarray) -> np.ndarray:
        """Returns a map that shows next bomb explosion for all fields"""
        board_size = len(board)
        danger_map = np.zeros(shape=(board_size, board_size), dtype=np.int)# + util.MAX_BOMB_LIFE
        while not bombs.empty():
            bomb = bombs.get()  # get bomb with lowest bomb life in queue
            # unpack tuple values
            bomb_life = bomb[0]
            bomb_range = bomb[1]
            bomb_pos = bomb[2]
            if danger_map[bomb_pos[0], bomb_pos[1]] and danger_map[bomb_pos[0], bomb_pos[1]] < bomb_life:
                # bomb already triggered by other bomb with shorter bomb_life, continue with next bomb in queue
                continue
            else:
                danger_map[bomb_pos[0], bomb_pos[1]] = bomb_life
            for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for dist in range(1, bomb_range):
                    r = bomb_pos[0] + row * dist
                    c = bomb_pos[1] + col * dist
                    if r < 0 or r >= board_size or c < 0 or c >= board_size:
                        # out of border
                        break
                    if bomb_blast_strength[r, c] != 0:
                        # we hit another bomb
                        if bomb_life < danger_map[r, c]:
                            bombs.put((bomb_life, int(bomb_blast_strength[r, c]), (r, c)))
                        break
                    elif board[r, c] in util.SOLID_TILES:
                        # solid tile stops bomb
                        danger_map[r, c] = min(danger_map[r, c] if danger_map[r, c] else 150, bomb_life)
                        break
                    else:
                        # update danger map
                        danger_map[r, c] = min(danger_map[r, c] if danger_map[r, c] else 150, bomb_life)
        return danger_map
