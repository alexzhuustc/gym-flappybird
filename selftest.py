import os, unittest
import logging
import time

import numpy as np
import bird_utils

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
class MyBirdTestCase(unittest.TestCase):
    def test_AggregateCounter(self):
        counter = bird_utils.AggregateCounter(10)
        for i in range(100):
            counter.append(i)

        self.assertEqual(counter.avg(), 94.5, "should be 94.5")

    def test_numpy_conditional(self):
        # 计算 y = a + [b > 0 ? c : 0]
        a = np.array(
            [
                [1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10]
            ]
        )

        b = np.array([
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1]
        ])

        c = np.ones((2,5))

        mask = b > 0
        y = a+ np.where(mask, c, 0)

        if True:
            logging.info("a is %s", a)
            logging.info("b is %s", b)
            logging.info("c is %s", c)
            logging.info("mask is %s", mask)
            logging.info("y is %s", y)

    def test_code(self):
        a =[1,2,3,4]
        print(a[:-2])
        pass

    def test_get_filename(self):
        name = max(os.listdir('logs/tensorboard'))
        logging.info('filename is %s', name)


    def test_replay_buffer(self):
        max_limit = 50* 10000
        insert_count = 100* 10000
        do_replay_buffer_baseline('deque', bird_utils.RandomReplayBuffer(max_limit), insert_count)
        do_replay_buffer_baseline('list', bird_utils.RandomReplayBuffer_ImplAsList(max_limit), insert_count)

def do_replay_buffer_baseline(name, replay_buffer, insert_count):
    #
    # 测试1
    #
    ts_begin = time.time()

    for i in range(insert_count):
        replay_buffer.append( (i,str(i)) )

    ts_elapsed = time.time() - ts_begin
    logging.info("1 %s initialize %s item takes %s seconds", name, insert_count, ts_elapsed)

    #
    # 测试2
    #
    ts_begin = time.time()
    sample_loop = 100
    sample_batch = 1024
    for loop in range(sample_loop):
        replay_buffer.sample(sample_batch)

    ts_elapsed = time.time() - ts_begin
    logging.info("2 %s sample %s item over %s times takes %s seconds", name, sample_batch, sample_loop, ts_elapsed)

    #
    # 测试3 添加很多数据
    #
    ts_begin = time.time()
    add_item_cnt = insert_count * 3
    for i in range(add_item_cnt):
        replay_buffer.append( (i,str(i)) )

    ts_elapsed = time.time() - ts_begin
    logging.info("3 %s add %s item takes %s seconds", name, add_item_cnt, ts_elapsed)

    #
    # 测试4
    #
    ts_begin = time.time()
    sample_loop = 100
    sample_batch = 1024
    for loop in range(sample_loop):
        replay_buffer.sample(sample_batch)

    ts_elapsed = time.time() - ts_begin
    logging.info("4 %s sample %s item over %s times takes %s seconds", name, sample_batch, sample_loop, ts_elapsed)

if __name__ == '__main__':
    unittest.main()
