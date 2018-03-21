
import os
import gym
from gym import wrappers
import cv2
import numpy as np
import scipy.misc
#from scipy.misc import imresize
from collections import deque
from dqn import DQN
import atari_wrappers
from atari_wrappers import make_atari,wrap_deepmind
from monitor import Monitor


reward_history = deque(maxlen=100)
t = 0
env = make_atari('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env,frame_stack=True)
#env = Monitor(env,'logs/experiment')
moves = env.action_space.n
#env = wrappers.Monitor(env,'/home/marcinic/hdd/projects/reinforcement/dqn/experiment-1',force=True)
agent = DQN((84,84),moves,priority_replay=False)


if os.path.exists("model/dqn.h5"):
    print("Loading model")
    nn = agent.load_model("model/dqn.h5")
    agent.epsilon = .1
else:
    nn = agent.init_model()

#nn = agent.init_model()

recent_frames = deque(maxlen=4)
target_update_freq = 10000

episodes = 100000
num_frames = 10000000000000

i = 0
total_reward = 0

for e in range(episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        if i<50000:
            action = env.action_space.sample()

        else:
            action = agent.act(state)
            #agent.epsilon_schedule(int(i)-50000)
        next_state,reward,done,info = env.step(action)


        agent.remember(state,action,reward,next_state,done)
        state = next_state
        episode_reward = episode_reward+reward
        i = i+4




        #total_reward = total_reward +episode_reward


        if(i % target_update_freq == 0):
            agent.update_target_model()

    reward_history.append(episode_reward)

    #total_reward = total_reward/max(1,e)
    #rh = np.array(reward_history)
    #if total_reward>max(rh):
    #    agent.best_network = agent.model

    if e%1000==0 and e>0:
        agent.report()
        avg = np.average(np.array(reward_history))
        print("Average 100 episode reward: "+str(avg))
        print("Average Q:"+ str(agent.avg_q))
        print("episode: {}/{}, score: {}, steps {}".format(e, episodes,episode_reward,i))
        print("Exploration:"+str(agent.epsilon))
        print("Memory length:"+str(agent.memory.__len__()))
    #if e%1000==0:
        print("Saving model")
        agent.model.save("model/dqn.h5")
