import multiprocessing as mp
import time
import uuid
import threading
import logging
from typing import Callable, Any



class RpcContext(object):
    def __init__(self):
        # 身份信息 master/slave
        self.my_role: str = None

        # 入参 (仅当my_role=slave)时有效
        self.args = None

        # 连接对点的PIPE
        self.pipe_to_another = None

        # 会话id (父子同享）
        self.session_id: str = None

        # 已启动的rpc_callback
        self.thread_rpc: threading.Thread = None

        # 供RPC线程检查，是否应该退出
        self.rpc_should_quit: bool = False

    def call_func_args(self, cmd_name, args):
        cmd = CustomCmd(cmd_name, args)
        self.call_cmd(cmd)

    def call_cmd(self, cmd):
        self.pipe_to_another.send(cmd)

    def call_cmd_quit(self):
        logging.info("send cmd QUIT to %s", self.session_id)
        self.stop_rpc_callback()
        self.call_cmd(SysCmd_Quit())

    def start_rpc_function_callback(self, rpc_callback):
        self.rpc_should_quit = False
        self.thread_rpc = threading.Thread(target=rpc_dispatch_proc, args=(self, None, rpc_callback)).start()

    def start_rpc_class_callback(self, class_instance, class_callback):
        self.rpc_should_quit = False
        self.thread_rpc = threading.Thread(target=rpc_dispatch_proc, args=(self,class_instance, class_callback)).start()

    def stop_rpc_callback(self):
        if self.thread_rpc:
            self.rpc_should_quit = True
            self.thread_rpc.join(timeout=3)
            self.thread_rpc = None


class MasterControl(object):

    global_session_id :int      = 0

    def __init__(self):
        pass

    def launch(self, proc, args) -> RpcContext:
        """
        启动子进程
        :param proc: 见示例slave_main_demo
        :param args: 出现在在ctx.args
        """
        pipe_parent, pipe_child = mp.Pipe()
        MasterControl.global_session_id += 1
        session_id = 'proc' + str(MasterControl.global_session_id)

        slave_ctx = RpcContext()
        slave_ctx.my_role = 'slave'
        slave_ctx.pipe_to_another = pipe_child
        slave_ctx.args = args
        slave_ctx.session_id = session_id

        p = mp.Process(target=proc, args=(slave_ctx,))
        p.start()

        master_context = RpcContext()
        master_context.my_role = 'master'
        master_context.pipe_to_another = pipe_parent
        master_context.session_id = session_id

        return master_context

class ProcessGroup(object):
    def __init__(self):
        pass


class SysCmd_Quit(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'SysCmd_Quit'

class CustomCmd(object):
    def __init__(self, cmd_name:str =None, args=None):
        self.cmd_name = cmd_name
        self.args = args

def rpc_dispatch_proc(ctx: RpcContext, class_instance, callback):
    while True:

        has_data = ctx.pipe_to_another.poll(1)
        if ctx.rpc_should_quit:
            return
        elif not has_data:
            continue


        try:
            cmd = ctx.pipe_to_another.recv()
        except EOFError as e:
            logging.info('%s %s, rpc pipe abort because: %s', ctx.my_role, ctx.session_id, 'EOFError')
            return

        try:
            if class_instance == None:
                callback(cmd)
            else:
                callback(class_instance, cmd)
        except Exception as e:
            logging.info('%s %s, encounter an exception %s', ctx.my_role, ctx.session_id, e)

        if isinstance(cmd, SysCmd_Quit):
            ctx.rpc_should_quit = True
            logging.info('%s %s, recv SysCmd_Quit, will quit', ctx.my_role, ctx.session_id)
            break


