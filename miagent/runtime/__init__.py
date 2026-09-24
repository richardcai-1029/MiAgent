"""多请求运行时：请求级并发、会话内串行、共享资源按优先级仲裁。

图内的并行是同一请求里彼此无依赖的任务同时执行；本包管的是图之上的一层
—— 多个请求同时在飞。纯 Python，不依赖图引擎。
"""

from .runtime import (INFERENCE_SLOTS, MAX_IDLE_SESSIONS, MAX_INFLIGHT, PRIORITY_RANK,
                      Runtime, build_runtime, dispatch_result)
from .slots import SlotPool, request_key

__all__ = ["INFERENCE_SLOTS", "MAX_IDLE_SESSIONS", "MAX_INFLIGHT", "PRIORITY_RANK", "Runtime", "SlotPool",
           "build_runtime", "dispatch_result", "request_key"]
