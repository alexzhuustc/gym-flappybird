import os
import bird_train
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
import logging
import numpy as np
import pygame
import sys
import types

# 启用下行，则禁用GPU，只使用CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

#MODEL_FILE = 'save_model/DQN_KERAS 20230925-STEP1000' # 很好的结果
MODEL_FILE = 'save_model/DQN_KERAS'

RENDER_MODE = 'human'      # render_mode='human' / None
ADD_NOISY_TO_ACTION = False

# Origin Method before Hijack
HOOKED_ORIGIN = {
    'draw_surface': None,
}

RUNNING = {
    'total_steps' : 0,
    'total_reward' : 0,
    'q_values' : [0, 0],
    'action' : 0,
}

def play():
    setting = bird_train.TrainingSetting()
    audio_on = True if RENDER_MODE else None
    env = gym.make(setting.ENV_NAME, render_mode=RENDER_MODE , audio_on=audio_on)
    logging.info("action space shape %s", env.action_space.shape)

    # hijack render() funtion
    try:
        hijack(env)
    except Exception as e:
        logging.error(e)
    #return

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

            RUNNING['total_steps'] = total_steps
            RUNNING['total_reward'] = total_reward
            RUNNING['q_values'] = result
            RUNNING['action'] = action


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

def hijack(env):
    # recursively wrapper
    env = env.env.env
    # analyze_object(env)

    renderer = env._renderer
    # analyze_object(renderer)

    HOOKED_ORIGIN['draw_surface'] = renderer.draw_surface

    # why types.MethodType? check links below
    # https://stackoverflow.com/questions/6650906/how-does-a-python-method-automatically-receive-self-as-the-first-argument
    renderer.draw_surface = types.MethodType(modified_draw_surface, renderer)

    # 补充初始化字体模块
    pygame.font.init()

def modified_draw_surface(self, show_score=True):
    if HOOKED_ORIGIN['draw_surface']:
        HOOKED_ORIGIN['draw_surface'](show_score)

    logging.info("modified_draw_surface ")
    RED = pygame.Color(255, 0, 0)
    BLUE = pygame.Color(0, 0, 255)
    GREEN = pygame.Color(0, 0, 255)
    # pygame.draw.circle(self.surface, RED, (100, 100), 50)
    font = pygame.font.Font(pygame.font.get_default_font(), 12)
    text_bitmap = font.render('gliding (no jump)', True, RED)
    text_rect = text_bitmap.get_rect()

    # 画出Jump vs NoJump区域
    surface_rect = self.surface.get_rect()
    self.surface.blit(text_bitmap, dest=(10, surface_rect.bottom - 10 - text_rect.height))

    delta = RUNNING['q_values'][0] - RUNNING['q_values'][1]
    abs_delta = int(abs(delta) * 8)
    if delta < 0:
        abs_delta = 0
    color_of_delta = RED if delta >= 0 else BLUE

    pygame.draw.rect(self.surface, color_of_delta, pygame.Rect(120, surface_rect.bottom - 9 - text_rect.height,abs_delta, 10))

    # Confidence
    qvalue_max = max(RUNNING['q_values'][0], RUNNING['q_values'][1])
    confidence = abs(int(qvalue_max/12.5 * 100 / 2))
    confidence_bitmap = font.render('confidence', True, RED)
    confidence_rect = confidence_bitmap.get_rect()
    self.surface.blit(confidence_bitmap, dest=(10, surface_rect.bottom - 30 - confidence_rect.height))
    pygame.draw.rect(self.surface, RED, pygame.Rect(120, surface_rect.bottom - 29 - text_rect.height, confidence, 10))

def analyze_object(obj):
    logging.info("analyze_object(%s)", obj)
    lines = []
    for attr_name in dir(obj):
        attr_type = 'attribute'
        if hasattr(obj, attr_name) and callable(getattr(obj, attr_name)):
            attr_type = 'method'

        lines.append(f' - {attr_type} {attr_name}')

    for line in sorted(lines):
        logging.info('%s', line)

def check_event():
    if RENDER_MODE:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    bird_train.init()
    play()
