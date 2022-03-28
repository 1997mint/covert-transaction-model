from SeqGAN.models import GeneratorPretraining, Discriminator, Generator,Discriminator_Cnn
from SeqGAN.utils import GeneratorPretrainingGenerator, DiscriminatorGenerator
from SeqGAN.rl import Agent, Environment
from keras.optimizers import Adam
import os
import numpy as np
import tensorflow as tf
sess = tf.Session()
import keras.backend as K
K.set_session(sess)

import code

class Trainer(object):
    '''
    Manage training
    '''
    def __init__(self, B, T, g_E, g_H, d_E, d_H, d_dropout, path_pos, path_neg, g_lr=1e-3, d_lr=1e-3, n_sample=16, generate_samples=10000, init_eps=0.1):
        self.B, self.T = B, T
        self.g_E, self.g_H = g_E, g_H
        self.d_E, self.d_H = d_E, d_H
        self.d_dropout = d_dropout
        self.generate_samples = generate_samples
        self.g_lr, self.d_lr = g_lr, d_lr
        self.eps = init_eps
        self.init_eps = init_eps
        self.top = os.getcwd()
        self.path_pos = path_pos
        self.path_neg = path_neg
        
        self.g_data = GeneratorPretrainingGenerator(self.path_pos, B=B, T=T, min_count=1) # next方法产生x, y_true数据; 都是同一个数据，比如[BOS, 8, 10, 6, 3, EOS]预测[8, 10, 6, 3, EOS]
        self.d_data = DiscriminatorGenerator(path_pos=self.path_pos, path_neg=self.path_neg, B=self.B, shuffle=True) # next方法产生 pos数据和neg数据

        self.V = self.g_data.V
        self.agent = Agent(sess, B, self.V, g_E, g_H, g_lr)
        self.g_beta = Agent(sess, B, self.V, g_E, g_H, g_lr)

        self.discriminator = Discriminator(self.V, d_E, d_H, d_dropout)

        self.discriminator_Cnn = Discriminator_Cnn(self.V,d_E,dropout = 0.4)

        self.env = Environment(self.discriminator, self.discriminator_Cnn,self.g_data, self.g_beta, n_sample=n_sample)

        self.generator_pre = GeneratorPretraining(self.V, g_E, g_H)

   

    def load_cnn(self, cnn_weight_path):
        self.discriminator_Cnn.load_weights(cnn_weight_path)

    

    def load(self, g_path, d_path):
        self.agent.load(g_path)
        self.g_beta.load(g_path)
        self.discriminator.load_weights(d_path)

    
    
    def generate_txt(self, file_name, generate_samples):
        path_neg = os.path.join(self.top, 'data', file_name)

        self.agent.generator.generate_samples(
            self.T, self.g_data, generate_samples, path_neg)

