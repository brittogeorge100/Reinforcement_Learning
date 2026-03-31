from padm_env import create_env
from q_learning import train_q_learning, visualize_q_table, run_the_game

train=True
visualize_results=True

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.3  # Minimum exploration  
epsilon_decay = 0.99  # Decay rate for exploration
no_episodes = 1000  # Number of episodes

space_invaders=[[5,1],[3,4],[7,3]]                  
space_shuttle=[[2,5],[6,9]]


if train:
    myenv = create_env(space_invaders=space_invaders, space_shuttle=space_shuttle)

    # Train a Q-learning agent:
    train_q_learning(myenv=myenv,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)
    
if visualize_results:
    # Visualize the Q-table:
    visualize_q_table(q_values_path="q_table.npy")

    #Play game with stored values from Q_table
    run_the_game(myenv=myenv)



'''
In this game with 15x9 grid cells, we have 3 Actions possible: Left, Right, Shoot. For Q learning, we create a 3x9 grid, where
these 3 rows correspond to each action: Row 0 ->Left, Row 1 ->Shoot, Row 2 ->Right. Once the agent is on Row 0, it can only 
move Left or shoot. Likewise in Row 2, it can only move Right or shoot. Only from Row 1 (state after shooting), Agent has the
possibility to move in any direction of choice.  
'''