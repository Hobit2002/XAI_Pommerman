"""An example to show how to set up an pommerman game programmatically"""
import pommerman
from agents.reflex.reflex_agent import ReflexAgent
from agents.heuristic.heuristic_agent import HeuristicAgent
from agents.learning.learning_agent import LearningAgent
import random, csv


def main():
    """Simple function to bootstrap a game."""
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Open log file
    csvfile =  open('game_records.csv', 'w', newline='')
    spamwriter = csv.writer(csvfile, delimiter=',')
    # Write the header
    header = ["game_id","step","player1_class","player2_class","player1_row","player1_col","player2_row","player2_col"]
    # Add columns describing the board
    for subject in ["board","bomb_blast_strength","bomb_life","flame_life"]:
        for r in range(11):
            for c in range(11):
                header.append(f"{subject}_r{r}_c{c}")
    spamwriter.writerow(header)
    game_id = 9


    # Create a set of agents
    for player_class,verbose_agent in [('monte_carlo',HeuristicAgent),('baby_depp_q',LearningAgent),('reflex',ReflexAgent)]:
        for enemy_class, enemy_agent in [('reflex',ReflexAgent), ('monte_carlo',HeuristicAgent), ('baby_depp_q',LearningAgent)]:
            agent_list = [
                verbose_agent(log = spamwriter, player_1 = player_class, player_2 = enemy_class, game_id = game_id),
                enemy_agent()
                ]
            # Make the "Free-For-All" environment using the agent list
            env = pommerman.make('PommeFFACompetition-v0', agent_list)

            # Run the episodes just like OpenAI Gym
            for i_episode in range(5):
                env._agents[0].update_game(game_id)
                state = env.reset()
                done = False
                while not done:
                    env.render()
                    actions = env.act(state)
                    state, reward, done, info = env.step(actions)
                try:
                    winner = env._agents[info["winners"][0]].__class__
                except KeyError:
                    winner = "Tie"
                print('Episode {} finished (Winner:{})'.format(i_episode,winner))
                game_id += 1
            env.close()


if __name__ == '__main__':
    main()
