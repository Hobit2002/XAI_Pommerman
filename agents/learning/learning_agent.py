import torch
import os
import pkg_resources
from typing import Dict, Any
from gym import spaces
import numpy as np
from typing import Dict, Any, Tuple, List

from pommerman import agents
from pommerman.constants import Item, Action

from . import net_input
from . import net_architecture
from agents.verbose_agent import VerboseAgent


# an example on how the trained agent can be used within the tournament
class LearningAgent(VerboseAgent):
    def __init__(self, *args, **kwargs):
        super(LearningAgent, self).__init__(*args, **kwargs)
        self.device = torch.device("cpu")  # you only have access to cpu during the tournament
        # place your model in the 'resources' folder and access them like shown here
        # change 'learning_agent' to the name of your own package (e.g. group01)
        model_file = os.path.join('agents','learning','resources', 'model.pt')

        # loading the trained neural network model
        self.model = net_architecture.DQN(board_size=11, num_boards=7, num_actions=6)
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.eval()

    def act(self, obs: Dict[str, Any], action_space: spaces.Discrete) -> int:
        # the learning agent uses the neural net to find a move
        # the observation space has to be featurized before it is fed to the model
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


        if self.log: self.log_state(obs)
        obs_featurized: torch.Tensor = net_input.featurize_simple(obs).to(self.device)
        with torch.no_grad():
            predictions: np.ndarray = self.model(obs_featurized).numpy()  # take highest rated move
            preferences = np.argsort(predictions[0])
            for i in range(len(preferences)):
                action = preferences[-i - 1]
                if action in actions: return action
        return 0
    
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
        return pos_val in [Item.Passage.value, Item.Kick.value,
                    Item.IncrRange.value, Item.ExtraBomb.value]
