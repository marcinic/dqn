
import os
import gym
from gym import wrappers
import cv2
import numpy as np
import scipy.misc
#from scipy.misc import imresize
from collections import deque
from dqn import DQN



def max_pixel(prev_frame,frame):
    """Takes maximum value for each pixel color of a frame"""
    return np.maximum(prev_frame,frame)

def reshape(frame):
    return cv2.resize(frame,(84,84),interpolation=cv2.INTER_AREA)
    #return imresize(frame,(84,84))

def extract_luminance(frame):
    """Extracts Y-channel from RGB frame. Reshapes to 84 by 84"""
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    frame = reshape(frame)
    return frame[:,:,None]


def huber_loss(x, clip_delta=1):
    error = np.abs(x)
    quadratic_part = np.minimum(error, clip_delta)
    return 0.5 * np.square(quadratic_part) + clip_delta * (error - quadratic_part)

episode_rewards = deque(maxlen=100)
t = 0
env = gym.make('BreakoutNoFrameskip-v4')
env.reset()
moves = env.action_space.n
#env = wrappers.Monitor(env,'/home/marcinic/hdd/projects/reinforcement/dqn/experiment-1',force=True)

agent = DQN((84,84),moves)


#if os.path.exists("model/dqn.h5"):
#    print("Loading model")
#    nn = agent.load_model("model/dqn.h5")
#else:
#    nn = agent.init_model()

nn = agent.init_model()

recent_frames = deque(maxlen=4)
target_update_freq = 10000

episodes = 100000
num_frames = 10000000000000

i = 0
for e in range(episodes):
    state = env.reset()
    state = extract_luminance(state)
    #es.append(state)
    reward = 0
    done = False
    action = 0
    time_t = 0
    total_reward = 0
    state_reward = 0
    prev_lives = 5
    for time_t in range(num_frames):

        #env.render()
        recent_frames.append(state)
        phi = np.concatenate(recent_frames,axis=2)

        # initialize with random agent
        if i < 50000:
            if i%4==0:
                action = env.action_space.sample()
            next_state, reward, done, info =  env.step(action)
            #reward = clip_reward(reward)
            state_reward+=reward

            #lives = info['ale.lives']
            #reward = clip_reward(reward)
            #if lives<prev_lives:
            #    reward = -10
            #    prev_lives = lives


            ns = recent_frames
            state = extract_luminance(next_state)
            ns.append(state)
            next_phi = np.concatenate(ns,axis=2)
            #next_phi = np.reshape(next_phi,(1,next_phi.shape[0],next_phi.shape[1],next_phi.shape[2]))
            total_reward = total_reward+reward
            agent.remember(phi.astype(np.int8),action,state_reward,next_phi.astype(np.int8),done)
            state_reward = 0
            i = i+1
        else:

            # Every 4 time steps select a move
            if time_t%4==0:
                phi = np.concatenate(recent_frames,axis=2)
                action = agent.select_arm(phi)
                next_state, reward, done, info =  env.step(action)


                state_reward +=reward
                 #if lives<prev_lives:
                #    reward = -10
                #    prev_lives = lives

                if not done: #np.count_nonzero(next_state)==0:
                    next_state = extract_luminance(next_state)
                    next_state = max_pixel(state,next_state)
                    ns = recent_frames
                    ns.append(next_state)
                    next_phi = np.concatenate(ns,axis=2)

                    agent.remember(phi.astype(np.int8),action,state_reward,next_phi.astype(np.int8),done)

                    #agent.update(rf,action,reward,ns,done)
                    state = next_state

                total_reward = total_reward+reward
                state_reward=0



            else:
                next_state, reward, done, info =  env.step(action)
                #reward = clip_reward(reward)
                state_reward += reward
            #    if lives<prev_lives:
            #        reward = -10
            #        prev_lives = lives

                total_reward = total_reward+reward


        i = i+1

        if done:
            #print("episode: {}/{}, score: {}, steps {}".format(e, episodes,total_reward,i))
            episode_rewards.append(total_reward)
            break
        if(i % target_update_freq == 0):
            #print("Updating target model")
            agent.update_target_model()
            agent.replay(agent.batch_size)

    #if len(agent.memory)>agent.batch_size:
    #        agent.replay(agent.batch_size)
    if e%100==0 and e>0:
        agent.report()
        avg = np.average(np.array(episode_rewards))
        print("Average 100 episode reward: "+str(avg))
        print("episode: {}/{}, score: {}, steps {}".format(e, episodes,total_reward,i))
    if e%1000==0:
        print("Saving model")
        agent.model.save("model/dqn.h5")


    #if len(agent.memory)>32 and (i % target_update_freq == 0) :
    #    agent.replay(32)
