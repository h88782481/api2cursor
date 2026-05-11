"""日志流式分发器

提供一个自定义的 logging 处理程序 (Handler)，用于将日志分发给多个活跃队列，
从而支持通过 SSE (Server-Sent Events) 机制将日志实时推送到前端管理面板。
"""

import logging
import queue
import threading


class SSELogHandler(logging.Handler):
    """将日志记录广播到所有活跃监听队列的处理器。"""

    def __init__(self, max_queue_size=200):
        super().__init__()
        # 存储所有活跃的监听队列
        self.listeners = []
        self.lock = threading.Lock()
        # 队列最大容量，防止单个监听器卡死导致内存溢出
        self.max_queue_size = max_queue_size

    def add_listener(self) -> queue.Queue:
        """创建一个新的监听队列并返回，用于 SSE 响应流。"""
        q = queue.Queue(maxsize=self.max_queue_size)
        with self.lock:
            self.listeners.append(q)
        return q

    def remove_listener(self, q: queue.Queue) -> None:
        """移除并注销指定的监听队列，通常在客户端断开连接时调用。"""
        with self.lock:
            if q in self.listeners:
                self.listeners.remove(q)

    def emit(self, record: logging.LogRecord) -> None:
        """核心回调：将格式化后的日志消息推送到所有在线的监听器中。"""
        try:
            msg = self.format(record)
            with self.lock:
                for q in self.listeners:
                    try:
                        # 采用非阻塞方式推送，如果某个队列已满，则跳过（丢弃该条日志）
                        q.put_nowait(msg)
                    except queue.Full:
                        pass
        except Exception:
            self.handleError(record)


# 实例化全局单例处理器
stream_handler = SSELogHandler()
# 设置默认的日志格式
stream_handler.setFormatter(
    logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
)
