import os
import shutil
import time
from collections import deque
import numpy as np
import random
import logging
import csv

def clear_content_in_directory(folder):
    """清空给定目录下的所有子目录和子文件。但保留文件夹自身"""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

class RandomReplayBuffer(object):
    """Experience replay buffer that samples uniformly."""

    def __init__(self, max_buffer_size):
        self.max_buffer_size = max_buffer_size
        self.buffer = deque(maxlen=max_buffer_size)

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def all_items(self):
        return self.buffer

    def append(self, item):
        self.buffer.append(item)

    def append_list(self, item_list):
        for item in item_list:
            self.append(item)

    def sample(self, batch_size):
        sample_data = []
        if self.size() == 0:
            return sample_data

        sample_indices = np.random.randint(0, len(self.buffer), size=batch_size)
        for s in sample_indices:
            sample_data.append(self.buffer[s])

        return sample_data

class RandomReplayBuffer_ImplAsList(object):
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

    def __len__(self):
        return self.size

    def size(self):
        return self.size

class AggregateCounter(object):
    '''即席聚合计算的指标容器'''
    def __init__(self, limit=None):
        self.limit = limit
        self.buffer = deque(maxlen=limit)
        self.value_avg = 0
        self.dirty_flag_avg = False

    def append(self, item) -> None:
        self.buffer.append(item)
        self.dirty_flag_avg = True

    def avg(self) -> float:
        if len(self.buffer) == 0:
            return 0

        if self.dirty_flag_avg:
            return self.__calc_avg()
        else:
            return self.value_avg

    def __calc_avg(self) -> float:
        self.dirty_flag_avg = False
        self.value_avg = sum(self.buffer) / len(self.buffer)
        return self.value_avg

class CooldownChecker(object):
    def __init__(self, interval : float = 1.0, allow_first : bool = False):
        '''
        :param interval: 冷却时间（秒）
        :param allow_first: 首次调用时，是否就绪。 True:就绪，False:未就绪
        '''
        self.interval = interval
        self.allow_first = allow_first
        self.total_trigger_count = 0
        self.last_trigger_time = time.time()

    def check(self) -> bool:
        if self.interval is None or self.interval < 0:
            return False

        if self.allow_first and self.total_trigger_count == 0:
            self.total_trigger_count += 1
            self.last_trigger_time = time.time()
            return True

        now = time.time()
        elapsed = now - self.last_trigger_time
        if elapsed >= self.interval:
            self.total_trigger_count += 1
            self.last_trigger_time = now
            return True

        return False

class SimpleCSV(object):
    def __init__(self, columns : list[str] = None):
        self.rows = []
        self.columns = columns
    def append_tuple(self, *args):
        row = list(args)

        if self.columns:
            if len(self.columns) != len(row):
                logging.info("SimpleCSV.append(), columns and args count are not match. columns is %s", self.columns)
                return

        kv = {}
        for idx, column in enumerate(self.columns):
            kv[column] =  args[idx]

        self.append_dict(kv)

    def append_kwargs(self, *kwargs):
        self.append_dict(kwargs)

    def append_dict(self, kv):
        self.rows.append(kv)

    def save(self, filename : str):
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=self.columns)
            csv_writer.writeheader()
            csv_writer.writerows(self.rows)
