import csv, time
import numpy as np
import pommerman
from pommerman import agents, characters
from group04 import group04_agent

#from . import characters

def get_board(row):
    board = np.array([int(item) for item in row[9:130]])
    return np.reshape(board, newshape=(11, 11))

def get_bombs(row):
    board = [item for item in row[9:130]]
    bombs = []
    for b, item in enumerate(board):
        if item == '3':
            row = b // 11
            col = b - 11 * row
            bombs.append(characters.Bomb(None, (row, col), 1,1))
    return bombs

def get_picture(row, env):
    env._board = get_board(row)
    env._bombs = get_bombs(row)
    env._flames = []#get_flames(row)
    env._step_count = 0
    env.render()


agent_list = [
        group04_agent.Group04Agent(),
        agents.SimpleAgent()
    ]
env = pommerman.make('PommeFFACompetition-v0', agent_list)

obtained_pictures = np.zeros(10)

with open("game_records_clustered.csv") as game_data:
    reader = csv.reader(game_data)
    for r,row in enumerate(reader):
        cluster = row[-1]
        if not r or min(obtained_pictures) != obtained_pictures[int(cluster)]: continue
        print(f"Showing example of cluster {cluster}. Press Ctrl + C when I should continue.")
        try:
            while True:
                get_picture(row, env)
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        obtained_pictures[int(cluster)] += 1