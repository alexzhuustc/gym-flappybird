import time
from collections import deque
import numpy as np
import logging
import csv
import tensorflow as tf
from tensorflow import keras

class DQN_Agent(object):
    # 模型
    model: tf.keras.Model
    target_model: tf.keras.Model

    def __init__(self):
        self.model = None
        self.target_model = None

    def build_model(self, observation_space, action_space, learning_rate=None) -> None:
        """初始化模型
        """

        self.model = keras.Sequential()

        self.model.add(keras.Input(shape=(observation_space,)))

        # fc1
        self.model.add(keras.layers.Dense(512, activation="relu"))

        # fc2
        self.model.add(keras.layers.Dense(256, activation="relu"))

        # A
        self.model.add(keras.layers.Dense(action_space, activation="linear"))

        # self.model.summary()

        # loss函数及训练方法
        self.model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

        # Double DQN需要一个额外模型
        # 构造一个Target
        self.target_model = keras.models.clone_model(self.model)
        self.copy_parameters_to_target_network()

    def copy_parameters_to_target_network(self):
        self.target_model.set_weights(self.model.get_weights())



def step_adjust(state_next, reward, is_terminated, is_truncated, info):
    #
    # 小鸟超过屏幕上沿，则认为失败
    #
    player_vertical_position = state_next[9]
    if player_vertical_position < 0:
        return state_next, -1.0, True, False, info

    #
    # 小鸟撞击地面而死，额外给个地面减分值 -1.0
    #
    if is_terminated and player_vertical_position > 0.7431:
        return state_next, reward - 1.0, is_terminated, is_truncated, info

    #
    # 小鸟撞击管子而死，额外给个距离减分值
    #
    first_pipe_x = state_next[0]
    if is_terminated and first_pipe_x > 0.3055:
        first_pipe_center = (state_next[1] + state_next[2]) / 2
        penalty = abs(player_vertical_position - first_pipe_center)
        return state_next, reward - penalty, is_terminated, is_truncated, info

    # 其它情况正常返回
    return state_next, reward, is_terminated, is_truncated, info