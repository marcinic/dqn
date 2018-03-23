
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





class DQN():
    def __init__(self,frame_dims,n_actions,priority_replay=True,epsilon=.99,discount=.99):
        self.epsilon_start = epsilon
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.final_exploration_frame = 1e6
        self.epsilon_decay = 0.99
        self.learning_rate = 0.00025
        self.discount = discount
        self.frame_dims = frame_dims
        self.n_actions = n_actions
        self.alpha = 0.7
        self.update_freq = 10000
        self.batch_size =32
        self.tb = TensorBoard(log_dir='./logs', write_graph=True,write_images=False)
        #self.summary_writer = K.summary.FileWriter('./logs/')
        self.beta = 0.5
        self.priority_replay_eps = 1e-6
        self.priority_replay = priority_replay
        self.avg_q=-1
        if priority_replay:
            self.memory = PrioritizedReplayBuffer(100000,self.alpha)
        else:
            self.memory = ReplayBuffer(600000)

    def init_model(self):
        #cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
        #    write_graph=True, write_images=True)

        inputs = Input(shape=(84,84,4,))
        normalized = keras.layers.Lambda(lambda x: x / 255.0)(inputs)
        conv_1 = Conv2D(32,kernel_size=(8,8),strides=(4,4),activation="relu",kernel_initializer="random_uniform")(normalized)
        conv_2 = Conv2D(64,kernel_size=(4,4),strides=(2,2),activation="relu",kernel_initializer="random_uniform")(conv_1)
        conv_3 = Conv2D(64,kernel_size=(3,3),strides=(1,1),activation="relu",kernel_initializer="random_uniform")(conv_2)
        flat = Flatten()(conv_3)
        dense_1 = Dense(512,activation="relu",kernel_initializer="random_uniform")(flat)
        output = Dense(self.n_actions,activation='linear')(dense_1)

        model = Model(inputs=inputs,outputs=output)
        opt = RMSprop(lr=self.learning_rate,rho=.95,epsilon=.01)#,clipnorm=1.)
        #opt = Adam(self.learning_rate)
        model.compile(optimizer=opt,loss=self.huber_loss)#,callbacks=[cb])
        self.model = model
        self.target_model = model


    def huber_loss(self,y_true, y_pred):
        return tf.losses.huber_loss(y_true,y_pred)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load_model(self,path):
        self.model = load_model(path,custom_objects={'huber_loss':self.huber_loss})
        self.target_model = self.model

    def act(self,state):
        state = np.reshape(state,(1,84,84,4))
        if random.random() < self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            probabilities = self.model.predict(state)
            action  = np.argmax(probabilities)

        self.replay(32)
        return action


    def clip_reward(self,reward):
        """Rewards are clipped to be between [-1,1]"""
        return np.sign(reward)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward,next_state, done)

    def make_targets(self,minibatch):
        """
        Creates targets equal to the reward if done is True
        and discounted future reward otherwise

        minibatch -- An nd-array of (s,a,r,ns,done) * batch size

        """

        actions = minibatch[1]


        Q = self.model.predict(minibatch[0])
        avg_q = np.mean(Q)
        self.avg_q = avg_q

        future = self.target_model.predict(minibatch[3])

        fv = np.amax(future,1)

        terminal = minibatch[4].astype(int)

        delta_o = minibatch[2] + (1-terminal)*self.discount*fv
        if self.priority_replay:
            q = np.zeros(self.batch_size)
            for i in range(0,self.batch_size):
                q[i] = Q[i][actions[i]]

            delta = q-delta_o
            td_error = abs(delta) +self.priority_replay_eps

            batch_idxes = minibatch[6]
            self.memory.update_priorities(batch_idxes,td_error)


        targets = Q 
        for i in range(0,self.batch_size):
            targets[i][actions[i]] = delta_o[i]


        return targets

    def q_learn_minibatch(self,batch_size):
        minibatch = []
        if self.priority_replay:
            minibatch = self.memory.sample(32,self.beta)
        else:
            minibatch = self.memory.sample(32)

        states = minibatch[0]

        targets = self.make_targets(minibatch)

        self.model.fit(states,targets,verbose=0)

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

    def epsilon_schedule(self,t):

        fraction = min(float(t)/self.final_exploration_frame,1.0)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_min-self.epsilon_start)

    def scale_frame(self,frame):
        return np.array(frame).astype(np.float32) / 255.0

    def replay(self, batch_size):
        self.q_learn_minibatch(batch_size)
