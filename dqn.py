
import keras
import random
import numpy as np
from collections import deque
from keras.layers import Input, Conv2D,Dense,Flatten
from keras.models import Model, load_model
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.callbacks import TensorBoard



class DQN():
    def __init__(self,frame_dims,n_actions,epsilon=1,discount=.99):
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.final_exploration_frame =1000000
        self.epsilon_decay = 0.99
        self.learning_rate = 0.00025
        self.discount = discount
        self.frame_dims = frame_dims
        self.n_actions = n_actions
        self.memory = deque(maxlen=10000)
        self.update_freq = 10000
        self.batch_size =32
        self.tb = TensorBoard(log_dir='./logs', write_graph=True,write_images=False)
        #self.summary_writer = K.summary.FileWriter('./logs/')


    def init_model(self):
        #cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
        #    write_graph=True, write_images=True)
        inputs = Input(shape=(84,84,4,))
        conv_1 = Conv2D(32,kernel_size=(8,8),strides=(4,4),activation="relu",kernel_initializer="random_uniform")(inputs)
        conv_2 = Conv2D(64,kernel_size=(4,4),strides=(2,2),activation="relu",kernel_initializer="random_uniform")(conv_1)
        conv_3 = Conv2D(64,kernel_size=(3,3),strides=(1,1),activation="relu",kernel_initializer="random_uniform")(conv_2)
        flat = Flatten()(conv_3)
        dense_1 = Dense(512,activation="relu",kernel_initializer="random_uniform")(flat)
        output = Dense(self.n_actions,activation='linear')(flat) #(dense_1)
        model = Model(inputs=inputs,outputs=output)
        opt = RMSprop(self.learning_rate,clipnorm=1.)
        #opt = Adam(self.learning_rate)
        model.compile(optimizer=opt,loss='mse')#,callbacks=[cb])
        self.model = model
        self.target_model = model



    def update_target_model(self):
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
            #print(y[1])

            Q = self.model.predict(np.reshape(y[0],(1,84,84,4)))
            #print(Q.shape)
            future = self.target_model.predict( np.reshape(y[3],(1,84,84,4)) )
            #print(future)
            fv = np.amax(future)
            max_action = np.argmax(future)


            #print(max_action==y[1])
            #print(fv)
            target =  y[2]+ (1-T)*(self.discount * fv)
            target = self.clip_reward(target)
            Q[0,y[1]] = target
            targets.append(Q)

        targets = np.array(targets).reshape((self.batch_size,self.n_actions))
        #print(targets[0].shape)thon
        return targets

    def q_learn_minibatch(self,batch_size):
        minibatch = random.sample(self.memory, batch_size)
        #print(minibatch[0][1].shape)
        #print(minibatch[0][0])
        states = np.array([state[0] for state in minibatch])
        #actions = np.array([state[1] for state in minibatch])
        #print(actions)
        states = states.reshape((batch_size,84,84,4))
        #print(states.shape)
        #next_states = np.array([ns[3] for ns in minibatch])
        #rewards = np.array([r[2] for r in minibatch])

        #q_values = self.model.predict(states)



        #action_q = np.array([q_value[action] for q_value,action in zip(q_values,actions)]) # Predicted q-value associated with action chosen in replay
        #print(action_q.shape)

        targets = self.make_targets(minibatch)

        #avg_q = targets.mean()#q_values.mean()
        #print("Average Q is "+str(avg_q))

        #print(targets)
        #fit_q = np.insert(q_values,actions,targets,axis=1)

        #for i in range(0,batch_size):
        #    q_values[i,actions[i]] = targets[i]

        #print(fit_q.shape)
        #loss = ((targets-action_q)**2).mean()

        #print(self.target_model.get_weights())
        self.model.fit(states,targets,verbose=0,callbacks=[self.tb]) #q_values)


        if self.epsilon>self.epsilon_min:
            self.epsilon *= self.epsilon_decay





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
