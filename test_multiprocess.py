import time
import unittest
import logging
from bird_multiprocess import RpcContext, MasterControl


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


def child_countdown_entry(ctx: RpcContext):
    logging.info("child_entry(), val is %s", ctx.args)
    slave = SlaveOp(ctx)

class MasterOp(object):
    def __init__(self, ctx : RpcContext):
        self.ctx: RpcContext = ctx

        # 启动监听线程
        ctx.start_rpc_class_callback(self, MasterOp.rpc_callback)

    def rpc_callback(self, cmd):
        logging.info("MasterClass(), recv cmd %s", cmd)


class SlaveOp(object):
    def __init__(self, ctx : RpcContext):
        self.ctx: RpcContext = ctx

        # 启动监听线程
        ctx.start_rpc_class_callback(self, SlaveOp.rpc_callback)

    def rpc_callback(self, cmd):
        logging.info("SlaveClass(), recv cmd %s", cmd)

def func_print(*args):
    logging.info("func_print, %s", args)

class MyBirdTestCase(unittest.TestCase):

    def test_mp_countdown(self):
        ctx = MasterControl().launch(child_countdown_entry, 4)
        master = MasterOp(ctx)

        master.ctx.call_cmd('hello')

        time.sleep(1)
        master.ctx.call_cmd_quit()


if __name__ == '__main__':
    unittest.main()
