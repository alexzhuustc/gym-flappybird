import os
import bird_ai
import bird_utils
import gymnasium as gym
import flappy_bird_gymnasium # noqa
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

MODEL_FILE = 'save_model/DQN_KERAS'

def play():
    env = gym.make('FlappyBird-v0', render_mode='human')

    model = keras.models.load_model(MODEL_FILE)

    play_details = bird_utils.SimpleCSV(['episode', 'bird_pos', 'bird_speed', 'bird_angle', 'pipe_dist', 'pipe_top', 'pipe_bottom', 'action', 'q_do_jump', 'q_do_nothing', 'reward', 'prev_state', 'new_state', 'is_terminated'])

    q_values_list = []
    actions = []

    episode = 0
    while True:
        episode += 1
        state, *info = env.reset()
        while True:
            prev_state = state
            q_values = model(np.array([prev_state]), training=False)

            result = list(q_values.numpy()[0])
            q_values_list.append(result)

            action = np.argmax(q_values[0], axis=-1)


            state, reward, is_terminated, is_truncated, info = env.step(action)
            state, reward, is_terminated, is_truncated, info = bird_ai.step_adjust(state, reward, is_terminated, is_truncated, info)

            play_details.append_tuple(episode, prev_state[9], prev_state[10], prev_state[11], prev_state[0], prev_state[1], prev_state[2],
                                      action, result[1], result[0], reward, prev_state, state, is_terminated)



            #logging.info("state: %s", state)
            #logging.info("q_values_list: %s", q_values_list)

            if is_terminated:
                play_details.save('logs/play_details.csv')
                return

            check_pygame_event()
            # time.sleep(0.1)


def check_pygame_event():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    play()
