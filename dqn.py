
import keras
import random
import numpy as np
from collections import deque
from keras.layers import Input, Conv2D,Dense,Flatten
from keras.models import Model, load_model
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.callbacks import TensorBoard
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
import tensorflow as tf

"""import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

    def init_torch(self):
        self.conv1 = nn.Conv2d(4,32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.fc4 = nn.Linear(7*7*64,512)
        self.fc5 = nn.Lineaer(512,self.n_actions)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0),-1)))
        return self.fc5(x)
"""




class DQN():
    def __init__(self,frame_dims,n_actions,epsilon=.1,discount=.99):
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.final_exploration_frame =1000000
        self.epsilon_decay = 0.99
        self.learning_rate = 0.00025
        self.discount = discount
        self.frame_dims = frame_dims
        self.n_actions = n_actions
        self.alpha = 0.7
        self.memory = PrioritizedReplayBuffer(10000,self.alpha)#deque(maxlen=100000)
        self.update_freq = 10000
        self.batch_size =32
        self.tb = TensorBoard(log_dir='./logs', write_graph=True,write_images=False)
        #self.summary_writer = K.summary.FileWriter('./logs/')
        self.beta = 0.5
        self.priority_replay_eps = 1e-6


    def init_model(self):
        #cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
        #    write_graph=True, write_images=True)
        inputs = Input(shape=(84,84,4,))
        conv_1 = Conv2D(32,kernel_size=(8,8),strides=(4,4),activation="relu",kernel_initializer="random_uniform")(inputs)
        conv_2 = Conv2D(64,kernel_size=(4,4),strides=(2,2),activation="relu",kernel_initializer="random_uniform")(conv_1)
        conv_3 = Conv2D(64,kernel_size=(3,3),strides=(1,1),activation="relu",kernel_initializer="random_uniform")(conv_2)
        flat = Flatten()(conv_3)
        dense_1 = Dense(512,activation="relu",kernel_initializer="random_uniform")(flat)
        output = Dense(self.n_actions,activation='linear')(dense_1)
        model = Model(inputs=inputs,outputs=output)
        #opt = RMSprop(self.learning_rate)#,clipnorm=1.)
        opt = Adam(self.learning_rate,clipnorm=1.)
        model.compile(optimizer=opt,loss=self.huber_loss)#,callbacks=[cb])
        self.model = model
        self.target_model = model


    def huber_loss(self,y_true, y_pred):
        err = y_true - y_pred
        HUBER_LOSS_DELTA = 2.0
        cond = K.abs(err) < HUBER_LOSS_DELTA
        L2 = 0.5 * K.square(err)
        L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

        loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

        return K.mean(loss)

    def update_target_model(self):
        #print(self.model.get_weights())
        #print(self.target_model.get_weights())
        self.target_model.set_weights(self.model.get_weights())

    def load_model(self,path):
        self.model = load_model(path)
        self.target_model = self.model

    def select_arm(self,state):
        state = np.reshape(state,(1,84,84,4))
        if random.random() < self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            probabilities = self.model.predict(state)
            action  = np.argmax(probabilities)

        self.replay(32)
        return action

    def perceive(self,state):
        state = np.reshape(state,(1,84,84,4))
        if random.random() < self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            probabilities = self.model.predict(state)
            action  = np.argmax(probabilities)
        return action

    def clip_reward(self,reward):
        """Rewards are clipped to be between [-1,1]"""
        return np.sign(reward)

    def remember(self, state, action, reward, next_state, done):
        if (state.shape==(84,84,4) and next_state.shape==(84,84,4)):
            self.memory.add(state, action, reward,next_state, done)
        else:
            pass

    def make_targets(self,minibatch):
        """
        Creates targets equal to the reward if done is True
        and discounted future reward otherwise

        minibatch -- An nd-array of (s,a,r,ns,done) * batch size

        """
        actions = minibatch[1]

        weights = minibatch[5]
        print(weights)
        Q = self.model.predict(minibatch[0])

        future = self.target_model.predict(minibatch[3]) #np.reshape(y[3],(1,84,84,4)) )

        fv = np.amax(future,1)

        terminal = minibatch[4].astype(int)

        delta_o = minibatch[2] + (1-terminal)*self.discount*fv

        q = np.zeros(self.batch_size)
        for i in range(0,self.batch_size):
            q[i] = Q[i][actions[i]]

        delta = q-delta_o
        #print(delta_o)
        delta = self.clip_reward(delta)
        #print(delta)

        #td_error = abs(delta) +self.priority_replay_eps
        #print(td_error)
        #batch_idxes = minibatch[6]
        #self.memory.update_priorities(batch_idxes,td_error)


        targets = Q #np.zeros((self.batch_size,self.n_actions))
        for i in range(0,self.batch_size):
            targets[i][actions[i]] = delta[i]
            #Q[i][actions[i]] = Q[i][actions[i]]+delta[i]

        return targets

    def q_learn_minibatch(self,batch_size):

        minibatch = self.memory.sample(32,self.beta)

        states = minibatch[0] #np.array([state[0] for state in minibatch])

        targets = self.make_targets(minibatch)

        self.model.fit(states,targets,verbose=0)#,callbacks=[self.tb]) #q_values)

        if self.epsilon>self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def report(self):
        layers = self.model.layers
        weights = self.model.get_weights()

        for i in range(len(weights)):
            weight_norms = np.mean(np.abs(weights[i]))

            print("Weight norms:")
            print(weight_norms)

            weight_max = np.abs(weights[i]).max()
            print("Weight max:")
            print(weight_max)

            weight_min = np.abs(weights[i]).min()
            print("Weight min:")
            print(weight_min)


    def replay(self, batch_size):
        self.q_learn_minibatch(batch_size)

    """    minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = state.reshape((1,84,84,4))
            next_state = next_state.reshape(1,84,84,4)
            target = reward
            if not done:
                target = reward + self.discount * \
                np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1,verbose=1)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
    """
