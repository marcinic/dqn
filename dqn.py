
import keras
import random
import numpy as np
from collections import deque
from keras.layers import Input, Conv2D,Dense,Flatten
from keras.models import Model
from keras.optimizers import RMSprop




class DQN():
    def __init__(self,frame_dims,n_actions,epsilon=.05,discount=.99,hist_len=1000):
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00025
        self.discount = discount
        self.frame_dims = frame_dims
        self.n_actions = n_actions
        self.hist_len = hist_len
        self.memory = deque(maxlen=1000)
        self.i = []


    def build_model(self):
        cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
          write_graph=True, write_images=True)
        S = Input(shape=(84,84,4,))
        h = Conv2D(16, 8, 8, subsample=(4, 4),
            border_mode='same', activation='relu')(S)
        h = Conv2D(32, 4, 4, subsample=(2, 2),
            border_mode='same', activation='relu')(h)
        h = Flatten()(h)
        h = Dense(256, activation='relu')(h)
        V = Dense(self.n_actions,activation='linear')(h)
        model = Model(S,V)
        rms = keras.optimizers.RMSprop(lr=0.00025)
        model.compile(optimizer=rms,loss='mean_squared_error',callbacks=[cb])
        self.model = model




    def init_model(self):
        cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
            write_graph=True, write_images=True)
        inputs = Input(shape=(84,84,4,))
        conv_1 = Conv2D(32,kernel_size=(8,8),strides=(4,4),activation="relu",kernel_initializer="glorot_uniform")(inputs)
        conv_2 = Conv2D(64,kernel_size=(4,4),strides=(2,2),activation="relu",kernel_initializer="glorot_uniform")(conv_1)
        conv_3 = Conv2D(64,kernel_size=(3,3),strides=(1,1),activation="relu",kernel_initializer="glorot_uniform")(conv_2)
        flat = Flatten()(conv_3)
        dense_1 = Dense(512,activation="relu",kernel_initializer="glorot_uniform")(flat)
        output = Dense(6)(dense_1)
        model = Model(inputs=inputs,outputs=output)
        model.compile(optimizer='RMSprop',loss='mean_squared_error',callbacks=[cb])
        self.model = model


    def select_arm(self,state):
        state = np.reshape(state,(1,84,84,4))
        if random.random() < self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            #action = values.argmax()
            probabilities = self.model.predict(state)
            #print(probabilities)
            self.model.fit(state,probabilities,epochs=1,verbose=0)
            #print(probabilities.shape)
            action  = np.argmax(probabilities)
        return action

    def update(self,state,action,reward,next_state,done):
        target = reward

        if not done:
            target = reward + self.discount * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)

        #loss = (target-np.amax(target_f))**2

        target_f[0][action] = target
        self.model.fit(state,target_f,epochs=1,verbose=0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward,next_state, done))


    def make_targets(self,transitions):
        """
        Creates targets equal to the reward if done is True
        and discounted future reward otherwise

        transitions -- A list of tuples (s,a,r,ns,done)

        """

        targets = []
        for y in transitions:
            T = 1
            if not y[4]:
                T = 0

            #print(type(ns))
            target =  y[2]+ (1-T)*(self.discount * np.amax(self.model.predict( np.reshape(y[3],(1,84,84,4)) )))
            #print(target)
            targets.append(target)

        targets = np.array(targets)
        #print(targets[0].shape)thon
        return targets

    def q_learn_minibatch(self,batch_size):
        minibatch = random.sample(self.memory, batch_size)
        #print(minibatch[0][1].shape)
        #print(minibatch[0][0])
        states = np.array([state[0] for state in minibatch])
        actions = np.array([state[1] for state in minibatch])
        #print(actions)
        states = states.reshape((batch_size,84,84,4))
        #print(states.shape)
        next_states = np.array([ns[3] for ns in minibatch])
        rewards = np.array([r[2] for r in minibatch])

        q_values = self.model.predict(states)
        action_q = np.array([q_value[action] for q_value,action in zip(q_values,actions)]) # Predicted q-value associated with action chosen in replay
        print(action_q)
        #print("Q values is "+str(q_values))
        targets = self.make_targets(minibatch)

        print(targets.shape)

        q_values[]
        loss = ((targets-action_q)**2).mean()
        self.model.fit(states,)
        #weights = self.model.trainable_weights
        #print(weights)
        #optimizer = RMSprop(0.00025)
        #updates = optimizer.get_updates(weights,[],loss)

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
