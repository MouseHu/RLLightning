import numpy as np
import gym
import time
from gym import error, spaces
from gym import core, spaces
from gym.envs.registration import register
import random
#from attn_toy.env.rendering import *
from attn_toy.env.fourrooms import Fourrooms,FourroomsNorender
from attn_toy.env.fourrooms_withcoin import FourroomsCoinNorender
from copy import copy,deepcopy
import cv2
from PIL import Image
import time
#plan:complicated noise
#random block that appears near the agent,but different from coin/goal/agent/wall
class FourroomsCoinNoiseKidNorender(FourroomsCoinNorender):
    def __init__(self, max_epilen=400, obs_size=128,seed=int(time.time())%1024):
        super(FourroomsCoinNoiseKidNorender, self).__init__(max_epilen,obs_size=obs_size,seed=seed)
        #self.background = np.zeros((2, obs_size, obs_size, 3),dtype=np.int)
        self.background = (np.random.rand(3,obs_size,obs_size,3)*64).astype(np.int);
        self.background[0, :, :, 1] += 32 # distinguish background
        self.background[1, :, :, 2] += 32  
        self.background[1, :, :, 0] += 32  
        #kid
        

    def render(self, state=-1):
        which_background = self.state % 3
        obs = deepcopy(self.background[which_background, ...])
        #add kid
        avail_states=deepcopy(self.init_states)
        if self.state in avail_states:
            avail_states.remove(self.state)
        kid = np.random.choice(avail_states)
        x, y = self.tocell[kid]
        blocks=[]
        if self.have_coin:
            blocks.append(self.make_block(x, y, (0, 0.5, 0.5)))
        else:
            blocks.append(self.make_block(x, y, (0.5, 0, 0.5)))
        #
        arr = self.render_origin(blocks)
        padding_height,padding_width = (obs.shape[0]-arr.shape[0])//2,(obs.shape[1]-arr.shape[1])//2
        obs[padding_height:padding_height+arr.shape[0],padding_width:padding_width+arr.shape[1],:] = arr-\
obs[padding_height:padding_height+arr.shape[0],padding_width:padding_width+arr.shape[1],:] 
        #background:[0,16]
        #obs:[0,255+16]

        new_obs=self.resize_obs(obs).astype(np.uint8)
        
        #start_p=(self.obs_size//2+10,self.obs_size//2+10)
        #end_p=(padding_height+arr.shape[0]-10,padding_width+arr.shape[1]-10)
        #im=cv2.cvtColor((new_obs),cv2.COLOR_BGR2RGB)
        #cv2.line(im,start_p,end_p,(0,127,0),2)
        #new_obs=np.array(im)
        return new_obs
    def resize_obs(self,obs):
        #resize observation array to [0,255]
        #input:(obs_size,obs_size,3)
        minimum=np.min(obs)
        obs_zero_min=obs-minimum
        maximum=np.max(obs_zero_min)
        #-1 to make sure maximum point <255
        return ((obs_zero_min*255)/maximum)-1
if __name__=='__main__':
    env=FourroomsCoinNoiseKidNorender()
    env.reset()
    cv2.imwrite('test0.jpg',env.render())
    env.step(0)
    cv2.imwrite('test1.jpg',env.render())
    cv2.imwrite('bgm2.jpg',env.background[1])
    cv2.imwrite('bgm1.jpg',env.background[0])
    env.step(3)
    cv2.imwrite('test2.jpg',env.render())
