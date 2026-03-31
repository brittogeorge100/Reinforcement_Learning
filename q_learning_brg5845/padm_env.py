import numpy as np
import gymnasium as gym
import pygame
import time

class MyEnv(gym.Env):
    def __init__(self, grid_size_x=9,grid_size_y=15) -> None:
        super(MyEnv, self).__init__()
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.cell_size = 60     
        self.position= None                     
        self.state = None
        self.alien=[]
        self.space_shuttle_list=[]
        self.reward=0
        self.info = {}
        self.done = False

        # Action-space:
        self.action_space = gym.spaces.Discrete(3)      #[0,1,2]
        
        # Observation space:
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([grid_size_x-1, grid_size_y-1]), dtype=np.int32)

        #We use pygame to set the screen
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size*self.grid_size_x, self.cell_size*self.grid_size_y))

    #Method 1: Reset
    def reset(self):
        self.position=np.array([4,self.grid_size_y-1])      #Position corresponds to real position in the game environment
        self.state=([4,1])                                  #states we are considering a 3x9 grid
        self.done = False
        self.reward = 0
        self.alien=[[5,1],[3,4],[7,3]]  
        self.space_shuttle_list=[[2,5],[6,9]]
        self.info["number of invader left"] = len(self.alien)
        
        return self.state, self.info
    
    #Method 2: create space_invaders and hell states
    def Space_invaders(self,space_invaders):              #Chitauri: positive reward
        self.alien=space_invaders

    def space_shuttle(self,space_shuttle):                #Thor: Negative reward
        self.space_shuttle_list=space_shuttle

    #Method 3: step() function
    def step(self,action):
            shoot=False

            if action==1 and self.position[0]>0:            #action 1: move left
                if self.state[1]==1:
                    self.state[1]=self.state[1]-1
                    self.state[0]=self.state[0]-1
                    self.position[0]=self.position[0]-1
                elif self.state[1]==0:
                    self.state[0]=self.state[0]-1
                    self.position[0]=self.position[0]-1
    
            elif action==2 and self.position[0]<8:          #action 2: move right
                if self.state[1]==1:
                    self.state[1]=self.state[1]+1
                    self.state[0]=self.state[0]+1
                    self.position[0]=self.position[0]+1
                elif self.state[1]==2:
                    self.state[0]=self.state[0]+1
                    self.position[0]=self.position[0]+1

            elif action==0:                                 #action 0: shoot
                shoot=True
                if self.state[1]==2:
                    self.state[1]=self.state[1]-1
                elif self.state[1]==0:
                    self.state[1]=self.state[1]+1
            
            #negative reward for every action Tony make
            self.reward=-.01

            #if shoot, shooting an alien removes alien & add +1 to reward
            while shoot:
                for i in self.alien:
                    if self.position[0]==i[0]:
                        self.reward=1
                        self.alien.remove(i)
                        break
            #If Thor is shot, a penalty of -1 is awarded
                if self.space_shuttle_list!=[]:
                    for i in self.space_shuttle_list:
                        if self.position[0]==i[0]:
                            self.space_shuttle_list.remove(i)
                            self.reward=-1
                shoot=False

            #If all the aliens are removed, agent recieves 10 point
            if self.alien==[]:
                self.done=True
                self.reward+=10
            self.info["number of invader left"] = len(self.alien)

            return self.state, self.reward, self.done, self.info      

    #Method 4 : Render()
    def render(self):
        pygame.init()
        screen=pygame.display.set_mode((self.cell_size*self.grid_size_x, self.cell_size*self.grid_size_y))

        #screen Background
        image = pygame.image.load("C11.jpg")
        image = pygame.transform.scale(image, (self.cell_size*self.grid_size_x, self.cell_size*self.grid_size_y))
        image.set_alpha(120)                #Fade background
        screen.blit(image, (0, 0))

        #draw grid lines
        for y in range(self.grid_size_y):
            for x in range(self.grid_size_x):
                grid=pygame.Rect(x*self.cell_size,y*self.cell_size,self.cell_size,self.cell_size)
                pygame.draw.rect(screen,(220,220,220), grid,1)          

        #Draw Chitauri
        alien_image = pygame.image.load("alien.jpg")
        alien_image = pygame.transform.scale(alien_image, (self.cell_size, self.cell_size))
        for i,n in enumerate(self.alien):
            alien_draw=pygame.Rect(n[0]*self.cell_size,n[1]*self.cell_size,self.cell_size,self.cell_size)    
            screen.blit(alien_image, alien_draw)                                                    

        #Draw Thor
        thor_image = pygame.image.load("thor.jpeg")
        thor_image = pygame.transform.scale(thor_image, (self.cell_size, self.cell_size))
        if self.space_shuttle_list!=[]:
            for i,n in enumerate(self.space_shuttle_list):
                thor=pygame.Rect(n[0]*self.cell_size,n[1]*self.cell_size,self.cell_size,self.cell_size)
                screen.blit(thor_image, thor)                                                             

        # Draw Tony:
        iron_image = pygame.image.load("iron.jpg")
        iron_image = pygame.transform.scale(iron_image, (self.cell_size, self.cell_size))
        agent = pygame.Rect(self.position[0]*self.cell_size, self.position[1]*self.cell_size, self.cell_size, self.cell_size)
        screen.blit(iron_image, agent)                                                                  
        
        # Update contents on the window:
        pygame.display.flip()

    # Method 4: .close()
    def close(self):
        pygame.quit()     

def create_env(space_invaders, space_shuttle):
    myenv=MyEnv(grid_size_x=9, grid_size_y=15)   
    myenv.Space_invaders(space_invaders)
    myenv.space_shuttle(space_shuttle)
    
    return myenv