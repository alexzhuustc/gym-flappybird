import os
import random

# 启用下行，则禁用GPU，只使用CPU
#   os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 启用下行，则TF不会把显存占满
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

import threading
from statistics import mean
from collections import deque
import gymnasium as gym
import flappy_bird_gymnasium # noqa
import time
import datetime
import logging
import numpy as np
from tensorflow import keras
import tensorflow as tf
import bird_utils
import bird_ai
import bird_multiprocess as bird_mp
import pickle
from typing import Any,Callable


class TrainingSetting(object):
    def __init__(self):
        # https://pypi.org/project/flappy-bird-gymnasium/
        self.ENV_NAME = "FlappyBird-v0"

        # 用于计算公式： 当前Q = 当下Reward + （未来Q * Gamma系数）
        self.GAMMA_FOR_FUTURE_Q_VALUE = 0.99

        # 最近一段时间平均步数达到此数值时，则退出
        self.QUIT_IF_RECENT_AVG_STEP_REACH_N = 5000

        # 训练时的AGENT个数
        self.AGENT_NUMBER = 5

        # Replay Buffer的大小
        # 以及触发训练的最小BufferSize。 （注：过小的Buffer容易过拟，所以不着急训练）
        self.REPLAY_BUFFER_SIZE = 500 * 1000
        self.NO_TRAIN_IF_REPLAY_BUFFER_SIZE_LESS_THAN = 10000

        # 训练时的学习率参数
        self.LEARNING_RATE = 0.0001

        # 训练时的FIT参数
        self.BATCH_SIZE = 1024 * 8
        self.MINI_BATCH_SIZE = 32

        # Double DQN target network的同步频率
        self.UPDATE_TARGET_NETWORK_AFTER_N_TRAINS = 25

        # 模型定时输出的位置
        self.DO_CHECKPOINT_EVERY_N_SECONDS = 60        # -1表示不保存
        self.CHECKPOINT_MODEL_FILE = 'save_model/DQN_KERAS'
        self.CHECKPOINT_REPLAY_BUFFER_FILE = None # 'logs/replay_buffer.data'

        # 是否从前一次训练中恢复模型参数 (None代表从头，文件名代表指定一个模型）
        self.DO_RESTORE = True
        self.RESTORE_MODEL_FILE = 'save_model/DQN_KERAS'
        self.RESTORE_REPLAY_BUFFER_FILE = None # 'logs/replay_buffer.data'

        # 运行时是否清理掉以前的tensorboard数据
        self.CLEAR_PREVIOUS_TENSORBOARD_DATA = True


class TrainingRuntime(object):
    def __init__(self, setting):
        # 训练的静态配置，不能修改
        self.setting : TrainingSetting  = setting

        # 训练的Agent
        self.agent : bird_ai.DQN_Agent = None

        # Replay Buffer
        self.replay_buffer : bird_utils.RandomReplayBuffer = None
        self.replay_buffer_lock = threading.Lock()

        # 累计流入的replay buffer个数
        self.replay_frame_input_count : int = 0
        self.replay_frame_train_count : int = 0

        # TensorBoard
        self.tensorboard_callback : keras.callbacks.TensorBoard = None
        self.metrics_writer : tf.summary.SummaryWriter = None

        # 当前已进行完成的游戏局数。 0代表即将玩首局
        self.episode: int = 0

        # 全局已训练过的次数
        self.training_count: int = 0

        # 模型更新过的次数
        self.model_update_count: int = 0

        # 最近50局的平均steps
        self.recent50_steps_lock = threading.Lock()
        self.recent50_steps: dict[str, float] = {}
        self.recent50_steps_avg : float = 0.0

        # q-value评估更新
        self.q_values_lock = threading.Lock()
        self.q_values = {}


    def setup(self):
        self.agent = bird_ai.DQN_Agent()
        self.replay_buffer = bird_utils.RandomReplayBuffer(self.setting.REPLAY_BUFFER_SIZE)

        #
        # 初始化tensorboard
        #
        tb_root = "logs/tensorboard/"
        tb_log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_logdir = tb_root + tb_log_name

        # 清理以前训练的数据
        if self.setting.CLEAR_PREVIOUS_TENSORBOARD_DATA:
            bird_utils.clear_content_in_directory(tb_root)

        # 可以启用，也可以不启用默认的tensorboard
        self.tensorboard_callback = None  # skeras.callbacks.TensorBoard(log_dir=tensorboard_logdir)

        # 我们写入自定义的metrics_writer
        self.metrics_writer = tf.summary.create_file_writer(logdir=tb_logdir + "/metrics", max_queue=200)

        #
        # 初始化模型
        #
        env = gym.make(self.setting.ENV_NAME, audio_on=False)
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        self.agent.build_model(observation_space, action_space, self.setting.LEARNING_RATE)

        if self.setting.DO_RESTORE:
            if self.setting.RESTORE_MODEL_FILE and os.path.exists(self.setting.RESTORE_MODEL_FILE):
                logging.info("restore weights from previous model %s", self.setting.RESTORE_MODEL_FILE)
                previous = keras.models.load_model(self.setting.RESTORE_MODEL_FILE)
                self.agent.model.set_weights(previous.get_weights())
                self.agent.target_model.set_weights(previous.get_weights())

            if self.setting.RESTORE_REPLAY_BUFFER_FILE and os.path.exists(self.setting.RESTORE_REPLAY_BUFFER_FILE):
                with open(self.setting.RESTORE_REPLAY_BUFFER_FILE, 'rb') as fin:
                    self.replay_buffer = pickle.load(fin)
                    logging.info("restore replay buffer from previous. file is %s, size is %s", self.setting.RESTORE_REPLAY_BUFFER_FILE, self.replay_buffer.size())

        # 默认tensorboard
        if self.tensorboard_callback:
            self.tensorboard_callback.set_model(self.agent.model)

    def metrics_add_scalar_value(self, name, value, interation = None):
        if interation is None:
            interation = self.training_count

        with self.metrics_writer.as_default(interation):
            tf.summary.scalar(name, data=value)

    def increase_training_count(self):
        self.training_count += 1

    def increase_module_update_count(self):
        self.model_update_count += 1

    def append_replay_buffer_deque(self, player: str, q: deque):
        try:
            self.replay_buffer_lock.acquire()
            for item in q:
                self.replay_buffer.append(item)

            self.replay_frame_input_count += len(q)
            # logging.info("replay buffer size changed to %s", self.runtime.replay_buffer.size())
        finally:
            self.replay_buffer_lock.release()

    def update_score(self, player: str, score: float):
        try:
            self.recent50_steps_lock.acquire()
            self.recent50_steps[player] = score
            self.recent50_steps_avg = mean(self.recent50_steps.values())
        finally:
            self.recent50_steps_lock.release()

    def update_q_value(self, player: str, q_value: list[float]):
        try:
            self.q_values_lock.acquire()
            self.q_values.update(q_value)
        finally:
            self.q_values_lock.release()

    def get_q_value(self, idxs: list[int]):
        ret = {}
        try:
            self.q_values_lock.acquire()
            for i in idxs:
                ret[i] = self.q_values.get(i)
        finally:
            self.q_values_lock.release()
        return ret


class Trainer(object):
    def __init__(self, setting: TrainingSetting, runtime: TrainingRuntime, globalsolver: 'GloablSolver'):
        self.setting = setting
        self.runtime = runtime
        self.globalsolver = globalsolver
        self.checkpoint_interval = bird_utils.CooldownChecker(self.setting.DO_CHECKPOINT_EVERY_N_SECONDS)
        self.model_broadcast_interval = bird_utils.CooldownChecker(1, True)
        self.fps_interval = bird_utils.CooldownChecker(5, True)

        self.prev_exp_replay_time = None
        self.prev_replay_frame_input_count = None
        self.prev_replay_frame_train_count = None

    def experience_replay_until_reach_goal(self):
        printsummary_cooldown = bird_utils.CooldownChecker(2, False)
        runtime = self.runtime
        setting = self.setting

        while True:

            if runtime.recent50_steps_avg and setting.QUIT_IF_RECENT_AVG_STEP_REACH_N:
                if runtime.recent50_steps_avg >= setting.QUIT_IF_RECENT_AVG_STEP_REACH_N:
                    logging.info("Reach goal %s > %s", runtime.recent50_steps_avg, setting.QUIT_IF_RECENT_AVG_STEP_REACH_N)
                    break

            fps_input = 0.0
            fps_train = 0.0
            if self.prev_exp_replay_time is not None:
                elpased = max(0.001, time.time() - self.prev_exp_replay_time)
                fps_input = (runtime.replay_frame_input_count - self.prev_replay_frame_input_count) / elpased
                fps_train = (runtime.training_count - self.prev_replay_frame_train_count) * setting.BATCH_SIZE / elpased

            if printsummary_cooldown.check():
                logging.info(
                    "Train Summary, fps input/train: %.02f/%.02f total train: %s, model update: %s, recent-50: %.02f, memory: %s",
                    fps_input,
                    fps_train,
                    runtime.training_count,
                    runtime.model_update_count,
                    runtime.recent50_steps_avg,
                    runtime.replay_buffer.size()
                )

            if runtime.replay_buffer.size() < setting.NO_TRAIN_IF_REPLAY_BUFFER_SIZE_LESS_THAN:
                time.sleep(0.1)
                continue

            if self.fps_interval.check():
                self.prev_exp_replay_time = time.time()
                self.prev_replay_frame_input_count = runtime.replay_frame_input_count
                self.prev_replay_frame_train_count = runtime.training_count

            self.experience_replay()


    def experience_replay(self):
        model = self.runtime.agent.model
        target_model = self.runtime.agent.target_model
        runtime = self.runtime
        setting = self.setting

        train_ts_start = time.time()
        sample_ts_start = time.time()
        verbose_mode = 0

        batch = runtime.replay_buffer.sample(setting.BATCH_SIZE)
        state_list, action_list, reward_list, next_state_list, terminated_list = zip(*batch)

        state = np.array(state_list)
        action = np.array(action_list)
        reward = np.array(reward_list)
        next_state = np.array(next_state_list)
        terminated = np.array(terminated_list)

        sample_elapsed = time.time() - sample_ts_start
        runtime.metrics_add_scalar_value('Train Sample Time', value=sample_elapsed)

        #
        # 以下是Double DQN算法
        #
        qvalue_ts_begin = time.time()

        batch_seq = np.arange(len(batch))

        # 1, 根据主模型推断出后续模型的最高行动项和最高价值
        best_actions = np.argmax(model(next_state, training=False), axis=-1)

        # 2, 把最佳action送入target network中，获取最佳action对应的q-value
        target_q_values = target_model(next_state, training=False).numpy()[batch_seq, best_actions]

        # 3, 主模型的q-values [current]
        q_current = model(state, training=False).numpy()
        q_expect = np.copy(q_current)

        # 4, 主模型的q-values [expect]
        #    如果terminated, 则不使用target数量，直接使用 reward
        q_adjusted = reward + np.where(terminated > 0, 0, setting.GAMMA_FOR_FUTURE_Q_VALUE * target_q_values)

        # 5, 最终结果
        q_expect[batch_seq, action] = q_adjusted

        # 6, 时
        qvalue_elapsed = time.time() - qvalue_ts_begin
        runtime.metrics_add_scalar_value('Train Prepare Q-Value Time', value=qvalue_elapsed)

        # 更新训练计步
        runtime.increase_training_count()

        #
        # 模式1，用FIT
        #
        train_ts_begin = time.time()
        history = model.fit(state, q_expect, batch_size=setting.MINI_BATCH_SIZE, verbose=verbose_mode)
        loss = tf.squeeze(history.history['loss'])

        #
        # 模式2，用train_on_batch。  注：搭配 tf.config.experimental.enable_op_determinism()
        #
        # loss = model.train_on_batch(state, q_expect)

        train_elapsed = time.time() - train_ts_begin
        runtime.metrics_add_scalar_value('Train Fit Time', value=train_elapsed)

        #
        # 输出loss到tensorboard
        #
        runtime.metrics_add_scalar_value('Train Q-value - Expect (train)', value=np.average(q_expect))
        runtime.metrics_add_scalar_value('Train Loss (train)', value=loss)

        #
        # 把明细训练数据输出为指标
        #
        q_value_before_adjust = q_current[batch_seq, best_actions]
        q_value_delta = q_adjusted - q_value_before_adjust
        runtime.metrics_add_scalar_value('Train Q-value - Delta (train)', value=np.average(q_value_delta))

        # 更新target network
        if runtime.training_count % setting.UPDATE_TARGET_NETWORK_AFTER_N_TRAINS == 0:
            runtime.agent.copy_parameters_to_target_network()

        # 广播变更信息
        if self.globalsolver:
            if self.model_broadcast_interval.check():
                runtime.increase_module_update_count()
                self.globalsolver.broadcast_model_update_event("model_" + str(time.time()), model.get_weights())

        # 耗时统计
        train_elapsed = time.time() - train_ts_start
        runtime.metrics_add_scalar_value('Train Total Time', value=train_elapsed)

        #
        # 输出其它来源指标，用于训练参考
        #
        if runtime.recent50_steps_avg > 0:
            runtime.metrics_add_scalar_value('Play Recent 50 Steps', value=runtime.recent50_steps_avg)


        for i, v in runtime.get_q_value([1, 50, 100, 150, 200, 250, 300]).items():
            if v == None:
                continue
            else:
                runtime.metrics_add_scalar_value('Play Q-Value at ' + str(i) + ' Step', value=v)


        # 保存checkpoint
        if self.checkpoint_interval.check():
            self.save_checkpoint()


    def save_checkpoint(self):
        agent = self.runtime.agent

        # save model
        if self.setting.CHECKPOINT_MODEL_FILE:
            agent.model.save(self.setting.CHECKPOINT_MODEL_FILE)
            logging.info('model has saved to %s', self.setting.CHECKPOINT_MODEL_FILE)

        # save replay buffer
        if self.setting.CHECKPOINT_REPLAY_BUFFER_FILE:
            with open(self.setting.CHECKPOINT_REPLAY_BUFFER_FILE, 'wb') as fout:
                try:
                    self.runtime.replay_buffer_lock.acquire()
                    pickle.dump(self.runtime.replay_buffer, fout)
                    logging.info('replay buffer has saved to %s', self.setting.CHECKPOINT_REPLAY_BUFFER_FILE)
                finally:
                    self.runtime.replay_buffer_lock.release()


'''
   0 the last pipe's horizontal position                [first]
   1 the last top pipe's vertical position
   2 the last bottom pipe's vertical position
   3 the next pipe's horizontal position                [second]
   4 the next top pipe's vertical position
   5 the next bottom pipe's vertical position
   6 the next next pipe's horizontal position           [third]
   7 the next next top pipe's vertical position
   8 the next next bottom pipe's vertical position
   9 player's vertical position
  10 player's vertical velocity
  11 player's rotation


  Reward
    +0.1 - every frame it stays alive
    +1.0 - successfully passing a pipe
    -1.0 - dying

  by https://github.com/markub3327/flappy-bird-gymnasium

'''
class Coordinator(object):
    def __init__(self, name, ctx: bird_mp.RpcContext, setting: TrainingSetting, runtime:TrainingRuntime):
        self.name                               = name
        self.ctx: bird_mp.RpcContext            = ctx
        self.setting: TrainingSetting           = setting
        self.runtime: TrainingRuntime           = runtime

        # 启动监听线程
        ctx.start_rpc_class_callback(self, Coordinator.rpc_callback)

    def rpc_callback(self, cmd):
        # logging.info("Coordinator(), from %s recv cmd %s", self.ctx.session_id, cmd)

        if isinstance(cmd, CmdUpdateReplayBuffer):
            c : CmdUpdateReplayBuffer = cmd
            session_id : str = self.ctx.session_id
            runtime = self.runtime

            runtime.append_replay_buffer_deque(session_id, c.replay_buffer.buffer)
            runtime.update_score(session_id, c.recent50_avg)
            runtime.update_q_value(session_id, c.q_values)

class Player(object):
    def __init__(self, ctx: bird_mp.RpcContext, startup: 'CmdStartup'):
       # hardcode setting
        self.env_name = startup.env_name
        self.replay_buffer_limit = startup.replay_buffer_limit
        self.replay_buffer_sendback_limit = startup.replay_buffer_sendback_limit

        # variables
        self.episode = 0
        self.step_in_episode = 0
        self.predict_count = 0
        self.q_values = {}
        self.model_version = None

        # 模型版本切换
        self.model_update_count = 0
        self.new_model_version = None
        self.new_model_weights = None
        self.new_model_lock = threading.Lock()

        # internal objects
        self.recent50_steps = bird_utils.AggregateCounter(50)
        self.agent = bird_ai.DQN_Agent()
        self.replay_buffer = bird_utils.RandomReplayBuffer(self.replay_buffer_limit)
        self.sendback_cooldown = bird_utils.CooldownChecker(1.0, True)
        self.playsummary_cooldown = bird_utils.CooldownChecker(30.0, False)

        env = gym.make(self.env_name, audio_on=False)
        self.agent.build_model(env.observation_space.shape[0], env.action_space.n)

        # 启动命令接收器
        self.ctx = ctx
        ctx.start_rpc_class_callback(self, Player.rpc_callback)

    def rpc_callback(self, cmd):
        # logging.info("Player(), recv cmd %s", cmd)

        if isinstance(cmd, CmdUpdateModelVersion):
            try:
                self.new_model_lock.acquire()
                self.new_model_version = cmd.model_version
                self.new_model_weights = cmd.model_weights
                self.model_update_count += 1
            finally:
                self.new_model_lock.release()

    def get_action(self, state):
        model = self.agent.model

        self.predict_count += 1
        q_values = model(np.array([state]), training=False)
        action = np.argmax(q_values[0], axis=-1)

        # 记录下平均q值
        if self.step_in_episode <= 1000:
            self.q_values[self.step_in_episode] = np.average(q_values)

        return action

    def get_random_exploring_action(self):
        #
        # 10%概率上跳
        #
        if np.random.rand() > 1 - 0.90:
            return 1
        else:
            return 0

    def should_do_random_exploring(self):
        # 玩的越好，ratio越低
        curr_steps = self.step_in_episode
        avg_steps = self.recent50_steps.avg()
        if avg_steps < 1:
            return True

        if avg_steps >= 1000:
            magic_ratio = 0.01
        elif avg_steps >= 100:
            magic_ratio = 0.05
        else:
            magic_ratio = 0.1

        base_ratio = min(magic_ratio, magic_ratio * 100 / avg_steps)

        if curr_steps < avg_steps - 100:
            rand_ratio = 0
        elif curr_steps < avg_steps - 50:
            rand_ratio = base_ratio / 10
        elif curr_steps < avg_steps - 20:
            rand_ratio = base_ratio / 5
        elif curr_steps < avg_steps:
            rand_ratio = base_ratio / 2
        else:
            rand_ratio = base_ratio

        return np.random.rand() > 1-rand_ratio

    def play_infinite(self):
        env = gym.make(self.env_name, audio_on=False)
        while True:
            #
            # episode开始
            #
            self.episode += 1
            self.step_in_episode = 0
            self.predict_count = 0
            self.q_values = {}

            episoda_begin_ts = time.time()
            total_reward = 0
            state, info = env.reset()

            # logging.info("player %s start episode %s", self.ctx.session_id, self.episode)

            while True:

                #
                # 是否收到退出信号
                #
                if self.ctx.rpc_should_quit:
                    logging.info("player %s receive QUIT", self.ctx.session_id)
                    return

                #
                # 是否有效模型
                #
                if self.model_version != self.new_model_version:
                    try:
                        self.new_model_lock.acquire()
                        if self.model_version != self.new_model_version:
                            # logging.info("agent model weight update %s -> %s", self.model_version, self.new_model_version)
                            self.agent.model.set_weights(self.new_model_weights)
                            self.model_version = self.new_model_version
                    finally:
                        self.new_model_lock.release()

                if self.model_version is None:
                    time.sleep(0.1)
                    continue
                #
                # 每步step开始
                #
                self.step_in_episode += 1

                #
                # 选择一个动作
                #
                if self.should_do_random_exploring():
                    action = self.get_random_exploring_action()
                else:
                    action = self.get_action(state)

                #
                # 更新状态
                #
                state_next, reward, is_terminated, is_truncated, info = env.step(action)
                state_next, reward, is_terminated, is_truncated, info = bird_ai.step_adjust(state_next, reward, is_terminated, is_truncated, info)

                #
                # 加入到Replay Buffer
                #
                item = (state, action, reward, state_next, is_terminated)
                self.replay_buffer.append(item)

                #
                # 计算total reward
                #
                total_reward += reward

                #
                # 下一个state
                #
                state = state_next

                #
                # 每局结束后，输出日志
                #
                if is_terminated:
                    # Recent 50
                    self.recent50_steps.append(self.step_in_episode)

                    # 打印日志
                    if self.playsummary_cooldown.check():
                        episoda_elapsed = max(0.000001, time.time() - episoda_begin_ts)
                        logging.info(
                                "Player %s episode: %s, model update: %s,  time: %.02f, fps: %.02f, steps in episode: %s, total_reward: %.02f, recent 50: %.02f, memory: %s",
                                self.ctx.session_id,
                                self.episode,
                                self.model_update_count,
                                episoda_elapsed,
                                self.step_in_episode / episoda_elapsed,
                                self.step_in_episode,
                                total_reward,
                                self.recent50_steps.avg(),
                                self.replay_buffer.size()
                        )

                    # 更新replay buffer
                    self.sendback_replay_buffer()
                    break


    def sendback_replay_buffer(self):
        if self.sendback_cooldown.check() == False:
            return

        cmd = CmdUpdateReplayBuffer()
        cmd.recent50_avg = self.recent50_steps.avg()
        cmd.q_values = self.q_values
        cmd.replay_buffer = self.replay_buffer
        self.replay_buffer = bird_utils.RandomReplayBuffer(self.replay_buffer_limit)

        if self.replay_buffer_sendback_limit:
            ''' prioritized replay buffer '''
            if self.replay_buffer.size() > self.replay_buffer_sendback_limit:
                prioritied_cnt = int(self.replay_buffer_sendback_limit / 2)
                trivial_cnt = self.replay_buffer_sendback_limit - prioritied_cnt
                all_frames = self.replay_buffer.all_items()

                self.replay_buffer = bird_utils.RandomReplayBuffer(self.replay_buffer_limit)
                self.replay_buffer.append_list(all_frames[-prioritied_cnt:])
                self.replay_buffer.append_list(random.sample(all_frames[:-prioritied_cnt], trivial_cnt))

        self.ctx.call_cmd(cmd)


class GloablSolver(object):
    def __init__(self):
        self.setting = TrainingSetting()
        self.runtime = TrainingRuntime(self.setting)
        self.trainer = Trainer(self.setting, self.runtime, self)
        self.coordinators = []

    def start_coordinators(self):
        for i in range(self.setting.AGENT_NUMBER):
            param = CmdStartup()
            param.env_name = self.setting.ENV_NAME
            param.replay_buffer_limit = self.setting.REPLAY_BUFFER_SIZE
            ctx = bird_mp.MasterControl().launch(remote_player_proc, param)
            coordinator = Coordinator('coordinator' + str(i), ctx, self.setting, self.runtime)
            self.coordinators.append(coordinator)

        for i in range(self.setting.AGENT_NUMBER):
            coordinator = self.coordinators[i]
            coordinator.ctx.call_cmd('hello')


    def stop_coordinators(self):
        for i,coordinator in enumerate(self.coordinators):
            coordinator.ctx.call_cmd_quit()

        self.coordinators = []

    def broadcast_model_update_event(self, version, weights):
        cmd = CmdUpdateModelVersion()
        cmd.model_version = version
        cmd.model_weights = weights
        for i in range(self.setting.AGENT_NUMBER):
            self.coordinators[i].ctx.call_cmd(cmd)

    def prepare_training(self):
        self.runtime.setup()

        logging.info("Training Logical Devices: %s", tf.config.list_logical_devices())
        logging.info("Training Physical Devices: %s", tf.config.list_physical_devices())
        gpus = tf.config.list_physical_devices('GPU')
        logging.info("GPUs Available: %s", gpus)

    def do_training_until_reach_goal(self):
        model = self.runtime.agent.model
        self.broadcast_model_update_event('first_model', model.get_weights())

        try:
            self.trainer.experience_replay_until_reach_goal()
        except KeyboardInterrupt as e:
            logging.info("user press ctrl+c, will abort")

class CmdStartup(object):
    def __init__(self):
        self.env_name : str = None
        self.replay_buffer_limit: int = None
        self.replay_buffer_sendback_limit: int = 100

    def __str__(self):
        return 'CmdStartup'

class CmdUpdateModelVersion(object):
    def __init__(self):
        self.model_version = None
        self.model_weights = None

    def __str__(self):
        return 'CmdUpdateModelVersion(version={0})'.format(self.model_version)

class CmdUpdateReplayBuffer(object):
    def __init__(self):
        self.replay_buffer : bird_utils.RandomReplayBuffer = None
        self.q_values = {}
        self.recent50_avg = 0

    def __str__(self):
        return 'CmdUpdateReplayBuffer(size={0})'.format(self.replay_buffer.size())

def remote_player_proc(ctx):
    name = ctx.session_id
    logging.basicConfig(format='%(asctime)s - ' + name + ' - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info("remote_player_proc()")
    logging.info("remote_player_proc()")
    logging.info("remote_player_proc()")

    # 仅使用CPU
    tf.config.set_visible_devices([], 'GPU')

    logging.info("remote_player_proc(), args are %s", ctx.args)
    player = Player(ctx, ctx.args)
    logging.info("remote_player_proc(), call play_infinite()")

    try:
        player.play_infinite()
    except KeyboardInterrupt as e:
        logging.info("user press ctrl+c, will abort")

def run():
    gserver = GloablSolver()

    try:
        gserver.prepare_training()
        gserver.start_coordinators()
        gserver.do_training_until_reach_goal()
        logging.info("trainer quit")

    finally:
        gserver.stop_coordinators()

def init():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True

if __name__ == "__main__":
    init()
    run()
