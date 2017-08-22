
import gym
import cv2
import numpy as np
from collections import deque
from replay_buffer import ReplayBuffer
from dqn import DQN

t = 0
env = gym.make('Breakout-v0')
env.reset()
agent = DQN((84,84),6)

nn = agent.init_model()


def max_pixel(prev_frame,frame):
    """Takes maximum value for each pixel color of a frame"""
    return np.maximum(prev_frame,frame)

def reshape(frame):
    return cv2.resize(frame,(84,84),interpolation=cv2.INTER_LINEAR) # bilinear interpolation


def extract_luminance(frame):
    """Extracts Y-channel from RGB frame. Reshapes to 84 by 84"""
    rs = reshape(frame)
    Y = cv2.cvtColor(rs,cv2.COLOR_RGB2GRAY)
    return Y



replay = ReplayBuffer(1000000)
recent_frames = deque(maxlen=4)
discount_factor = 0.99
experience = []
#total_reward = 0.0
episodes = 100
num_frames = 10000

i = 0
for e in range(episodes):
    state = env.reset()
    state = extract_luminance(state)
    #recent_frames.append(state)
    reward = 0
    done = False
    action = 0
    time_t = 0
    total_reward = 0
    for time_t in range(num_frames):

        #env.render()
        recent_frames.append(state)
        phi = np.stack(recent_frames,axis=0)

        # initialize with random agent
        if i < 10000:
            if i%4==0:
                action = env.action_space.sample()
            next_state, reward, done, info =  env.step(action)
            ns = recent_frames
            state = extract_luminance(next_state)
            ns.append(state)
            next_phi = np.stack(ns,axis=0)
            #next_phi = np.reshape(next_phi,(1,next_phi.shape[0],next_phi.shape[1],next_phi.shape[2]))
            total_reward = total_reward+reward
            agent.remember(phi,action,total_reward,next_phi,done)

            i = i+1
        else:


            if time_t%4==0:
                phi = np.stack(recent_frames,axis=0)
                action = agent.select_arm(phi)
                next_state, reward, done, info =  env.step(action)
                if not done: #np.count_nonzero(next_state)==0:
                    next_state = extract_luminance(next_state)
                    next_state = max_pixel(state,next_state)
                    ns = recent_frames
                    ns.append(next_state)
                    next_phi = np.stack(ns,axis=0)
                    #replay.add(rf,action,reward,ns,done)
                    agent.remember(phi,action,reward,next_phi,done)

                    #agent.update(rf,action,reward,ns,done)
                    state = next_state

                total_reward = total_reward+reward



            else:
                next_state, reward, done, info =  env.step(action)
                total_reward = total_reward+reward





        if done:
            print("episode: {}/{}, score: {}".format(e, episodes,total_reward))
            break
    agent.replay(32)
