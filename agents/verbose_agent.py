
from typing import Dict, Any, Tuple, List
import numpy as np

from pommerman.constants import Item
from pommerman import agents

class VerboseAgent(agents.BaseAgent):

    def __init__(self, *args, **kwargs):
        if "log" in kwargs.keys():
            self.log = True
            self.writer = kwargs["log"]
            self.player_1 = kwargs["player_1"]
            self.player_2 = kwargs["player_2"]
            self.game_id = kwargs["game_id"]
            del kwargs["log"]
            del kwargs["player_1"]
            del kwargs["player_2"]
            del kwargs["game_id"]
        else: self.log = False
        super(VerboseAgent, self).__init__(*args, **kwargs)

    def log_state(self,obs):
        enemy: Item = obs['enemies'][0]  # we only have to deal with 1 enemy
        epos: np.ndarray = np.where(obs['board'] == enemy.value)
        enemy_position: Tuple[int, int] = [epos[0][0], epos[1][0]]
        self.writer.writerow(
            [self.game_id, obs["step_count"], self.player_1,self.player_2] + list(obs["position"]) + enemy_position + list(obs["board"].flatten()) + list(obs["bomb_blast_strength"].flatten()) + list(obs["bomb_life"].flatten()) + list(obs["flame_life"].flatten()) 
        )

    def update_game(self,new_id):
        self.game_id = new_id