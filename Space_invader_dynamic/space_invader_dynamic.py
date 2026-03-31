import random
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pygame
import time


x_grid=int(9)                                                                                                                                               
y_grid=int(20)
game_space=np.array([x_grid,y_grid])                
current_position=np.array([4,y_grid-1])                                                     #
cell_size=30                                                                                #
score=0
alien=[]                                                                                    #
pygame.init()                                                                               #
screen=pygame.display.set_mode((cell_size*x_grid, cell_size*y_grid))                        #



def initialize_reward_state() ->tuple:
    num_states=18
    actions=[-1,0,1]
    transition_probs = np.zeros((num_states, len(actions), num_states))
    success_prob = 0.8
    fail_prob = 0.2
    #shoot=False

    for s in range(num_states):
        for a in actions:
            intended_state=s
            if a==-1 and s>1 and s%2==0:
                intended_state=s-2
                #shoot=False
            elif a==-1 and s>1 and s%2==1:
                intended_state=s-3
                #shoot=False
            elif a==1 and s<16 and s%2==0:
                intended_state=s+2
                #shoot=False
            elif a==1 and s<16 and s%2==1:
                intended_state=s+1
                #shoot=False
            elif a==0 and s%2==0:
                intended_state=s+1
                #shoot=True
            if s%2==1 and a==0:
                 transition_probs[s,0,:]=0
            else:
                transition_probs[s,a,s]=fail_prob
                transition_probs[s,a,intended_state]=success_prob
    return transition_probs

def extract_policy(V, rewards, transition_probs, gamma: float = 0.9):
    
    num_states = len(V)
    #print(num_states)
    policy = np.zeros(num_states, dtype=int)

    for s in range(num_states):
        best_action = None
        best_value = float('-inf')
        for a in [-1,0,1]:  # Actions are [-1,0,1]
            sum_value = 0
            for s_prime in range(num_states):
                sum_value += transition_probs[s][a][s_prime] * (rewards[s][a][s_prime] + gamma * V[s_prime])
                # if s==0:
                #     print('value of', a, 'is:', sum_value)
            if sum_value > best_value:
                best_value = sum_value
                best_action = a
        policy[s] = best_action

    return policy

def run_value_iteration(gamma:float=0.9, threshold: float=0.01):
    transition_probs=initialize_reward_state()
    states=list(range(18))
    actions=[-1, 0, 1]

    rewards=np.full((len(states), len(actions),len(states)),-1)

    #shooting from an already shot point, without any invaders
    for i in range(18):
        if i%2==1:
            rewards[i,0,:]=-10

    #shooting an invader earns point, while without invader earns negative point
    for i in range(9):
        if i in [x[0] for x in alien]:
            rewards[(i*2),0,:]=10
        else:
            rewards[(i*2),0,:]=-50
    
    #shooting the invader that is about to fall
    close_call_present=False
    '''
    alien_copy=alien.copy()
    alien_sorted_list=sorted(alien_copy, key=lambda x:x[1])
    for i in alien_sorted_list:
        steps_left=20-i[1]
        remaining_steps=steps_left-current_position[0]
        alien_delete=alien_sorted_list.copy()
    '''
    for i in alien:
        steps_left=20-i[1]
        remaining_steps=steps_left-(abs(i[0]-current_position[0]))
        alien_delete=alien.copy()

        alien_delete.remove(i)
        for j in alien_delete:
            if j[0]!=i[0]:
                if remaining_steps<2:
                    close_call_present=True
                    rewards[(j[0]*2),0,:]-=12
        if close_call_present:
            close_call_present=False
            break
    
    
    V=np.zeros(len(states))
    count=0
    while True:
        delta=0
        for s in states:
            v = V[s]
            new_v = max([
                sum(
                    transition_probs[s][a][s_prime] *
                        (rewards[s][a][s_prime] + gamma * V[s_prime])
                        for s_prime in states)
                for a in actions])

            V[s] =new_v
            delta=max(delta,abs(v-V[s]))

        count+=1
        if delta < threshold:
            break
        #print(f"V(s) for iteration: {count} \n", V, "\n")
        #print(f"R(s) for iteration: {count} \n", rewards, "\n")    
    
    policy = extract_policy(V, rewards, transition_probs, gamma)
        
    print('policy is: ',policy)
    return policy


def game():
    game_continue=True
    shoot=False
    score=0
    count=0

    alien_used=set()                                #
    for i in range(3):                              #
        #alien_start=[random.randint(0,8),0]
        while True:                                 #
            k=random.randint(0,8)
            if k not in alien_used:                 #
                alien_used.add(k)                   #
                break                               #
        alien_start=[k,0]                           #
        alien.append(alien_start)                   #

    while game_continue:
        count+=1
        print(count)
        policy=run_value_iteration()
        render(screen)
        if policy[(2*current_position[0])]==-1 and current_position[0]>0:
            current_position[0]=current_position[0]-1
            shoot=False
        elif policy[(2*current_position[0])]==1 and current_position[0]<8:
            current_position[0]=current_position[0]+1
            shoot=False
        elif policy[(2*current_position[0])]==0:
            shoot=True
        
        for i in range(len(alien)):
            alien[i][1]+=1
            score-=1
        while shoot:
            for i in alien:
                if current_position[0]==i[0]:
                    score+=10
                    alien.remove(i)
                    alien_start=[random.randint(0,8),0]
                    alien.append(alien_start) 
                    break
            shoot=False       
        if alien==[]:
            print('you won')
            game_continue=False
        for i in alien:
            if i[1]==y_grid-1:
                print('you suck')
                score-=100
                game_continue=False
        time.sleep (1)
    

def render(screen):
    pygame.init()
    screen=pygame.display.set_mode((cell_size*x_grid, cell_size*y_grid))

    #fill screen
    screen.fill((255,255,255))

    #draw grid lines
    for y in range(y_grid):
        for x in range(x_grid):
            grid=pygame.Rect(x*cell_size,y*cell_size,cell_size,cell_size)
            pygame.draw.rect(screen,(200,200,200), grid,1)

    #draw aliens
    for i,n in enumerate(alien):
        alien_draw=pygame.Rect(n[0]*cell_size,n[1]*cell_size,cell_size,cell_size)
        pygame.draw.rect(screen, (0,255,0), alien_draw)

    # Draw the agent:
    agent = pygame.Rect(current_position[0]*cell_size, current_position[1]*cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, (0,0,0), agent)

     # Update contents on the window:
    pygame.display.flip()
             
def close():
    pygame.quit()

game()
#run_value_iteration()