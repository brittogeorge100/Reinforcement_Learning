import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Function 1: Train Q-learning agent
def train_q_learning(myenv,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):
    
    q_table = np.zeros((myenv.grid_size_x,3, myenv.action_space.n))         # (9x3x3 table: X state, Y state, action correspondingly)
    # Q-learning algorithm:
    for eps in range(no_episodes):
        state, _ = myenv.reset()
        state = tuple(state)   
        total_reward = 0.0

        while True:
            if epsilon_min<epsilon:
                action = myenv.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit
            next_state, reward, done, _ = myenv.step(action)
            #myenv.render()                                         
            next_state = tuple(next_state)
            total_reward += reward
            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma *
                 np.max(q_table[next_state]) - q_table[state][action])
            state = next_state

            if done:
                myenv.close()
                break
        #epsilon decay to change from exploration to exploitation 
        epsilon=epsilon*epsilon_decay

    myenv.close()
    print("Training finished.\n")

    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")


# Function 2: Visualize the Q-table
def visualize_q_table(actions=["shoot", "Left", "Right"],
                      q_values_path="q_table.npy"):
    try:
        q_table = np.load(q_values_path)

        _, axes = plt.subplots(1, 3, figsize=(15, 3))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:,:, i].copy()
            heatmap_data = heatmap_data. T
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, annot_kws={"size": 9})
            ax.set_title(f'Action: {action}')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")

#Function 3: function will run the game with values from q_table
def run_the_game(myenv,
                 q_values_path="q_table.npy"):
    state,_ = myenv.reset()
    total_point=0
    while True:
        myenv.render()
        q_table = np.load(q_values_path)
        state=tuple(state)
        action=np.argmax(q_table[state])
        next_state, reward, done, _ = myenv.step(action)
        total_point+=reward
        state = next_state
        time.sleep(1)
        if done:
            print('you won, point: ')
            myenv.close()
            break
