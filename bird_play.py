import os
import bird_train
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
import logging
import numpy as np
import pygame
import sys

# 启用下行，则禁用GPU，只使用CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

#MODEL_FILE = 'save_model/DQN_KERAS 20230925-STEP1000' # 很好的结果
MODEL_FILE = 'save_model/DQN_KERAS'

RENDER_MODE = 'human'      # render_mode='human' / None
ADD_NOISY_TO_ACTION = False

def play():
    setting = bird_train.TrainingSetting()
    audio_on = True if RENDER_MODE else None
    env = gym.make(setting.ENV_NAME, render_mode=RENDER_MODE , audio_on=audio_on)
    logging.info("action space shape %s", env.action_space.shape)

    model = keras.models.load_model(MODEL_FILE)
    q_values_list = []
    actions = []
    while True:
        total_reward = 0
        total_steps = 0
        state, *info = env.reset()
        while True:
            q_values = model(np.array([state]), training=False)

            result = q_values.numpy()[0]
            q_values_list.append(list(result))

            action = np.argmax(q_values[0], axis=-1)
            if ADD_NOISY_TO_ACTION:
                if np.random.rand() > 0.95:
                    if np.random.rand() > 0.9:
                        action = 1
                    else:
                        action = 0
                    logging.info("action noisy. new action is %s", action)

            actions.append(action)


            state, reward, is_terminated, is_truncated, *info = env.step(action)
            total_steps += 1
            total_reward += reward

            #logging.info("state: %s", state)
            #logging.info("q_values_list: %s", q_values_list)

            if is_terminated:
                break

            check_event()
            # time.sleep(0.1)

        # logging.info("actions: %s", actions)
        logging.info("total steps: %s, total reward: %s", total_steps, total_reward)

def check_event():
    if RENDER_MODE:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    bird_train.init()
    play()
