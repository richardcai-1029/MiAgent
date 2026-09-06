"""stdio 传输层：在两个进程之间收发 JSON-RPC 报文。

端侧选 stdio 而不是 HTTP 的理由：
  · 零依赖 —— Python 内置，不需要 HTTP 服务器和客户端库
  · 不占端口，也就没有端口冲突和本机网络暴露面
  · 父进程退出时子进程被自动回收，生命周期天然可控

分帧（framing）约定：一条消息一行，行尾 \\n。
JSON 序列化会把内容里的换行转义成 \\\\n，所以正文里的换行不会破坏分帧。

⚠️ 服务端进程的 stdout 是数据通道。任何 print() 都会插进协议流里让对端解析失败。
   服务端要输出日志，一律用本模块的 log()，它写 stderr。
"""

from __future__ import annotations

import json
import select
import subprocess
import sys
from typing import IO, Any

from .protocol import ErrorCode, MiClawError


def log(*args: Any) -> None:
    """服务端专用日志。必须走 stderr —— stdout 被协议占用了。"""
    print(*args, file=sys.stderr, flush=True)


# ============================================================
# 分帧读写：整个传输层就靠这两个函数
# ============================================================


def write_message(stream: IO[str], msg: dict[str, Any]) -> None:
    """把一条报文写进流。"""
    # separators 去掉多余空格，端侧能省一点带宽和内存
    line = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
    stream.write(line + "\n")
    # flush 不能省！管道是带缓冲的，不 flush 消息会卡在缓冲区里，
    # 对端一直读不到 —— 双方互相等待，就是死锁。
    stream.flush()


def read_message(stream: IO[str]) -> dict[str, Any] | None:
    """从流里读一条报文。返回 None 表示对端已关闭（EOF）。"""
    while True:
        line = stream.readline()
        if line == "":            # 空字符串 = EOF；空行是 "\n"，两者不同
            return None
        line = line.strip()
        if not line:              # 跳过空行，容忍对端多打了换行
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise MiClawError(
                ErrorCode.MC_INVALID_MESSAGE,
                "报文不是合法 JSON",
                detail={"line": line[:200], "reason": str(e)},
            ) from e


# ============================================================
# 两侧的传输端点
# ============================================================


class StdioServerTransport:
    """服务端侧：从自己的 stdin 读请求，往自己的 stdout 写响应。"""

    def __init__(self, stdin: IO[str] | None = None, stdout: IO[str] | None = None):
        self._in = stdin or sys.stdin
        self._out = stdout or sys.stdout

    def receive(self) -> dict[str, Any] | None:
        return read_message(self._in)

    def send(self, msg: dict[str, Any]) -> None:
        write_message(self._out, msg)


class SubprocessTransport:
    """客户端侧：拉起服务端子进程，通过它的管道通信。

    子进程的 stderr 不接管，直接继承父进程的 —— 这样服务端的日志会
    出现在我们终端里，调试时能看到两边在干什么。
    """

    def __init__(self, command: list[str]):
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,  # 行缓冲，配合我们的按行分帧
        )

    def send(self, msg: dict[str, Any]) -> None:
        if self._proc.poll() is not None:
            raise MiClawError(
                ErrorCode.MC_TRANSPORT_CLOSED,
                "服务端进程已退出",
                detail={"returncode": self._proc.returncode},
            )
        write_message(self._proc.stdin, msg)

    def receive(self, timeout: float | None = None) -> dict[str, Any] | None:
        """读一条响应。timeout 为秒；超时抛 MC-1003。

        注：select 在 POSIX 上能作用于管道（我们的目标平台是 Android/Linux），
        Windows 上 select 只支持 socket，那边需要换成读线程 + 队列。
        """
        if timeout is not None:
            ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
            if not ready:
                raise MiClawError(
                    ErrorCode.MC_REQUEST_TIMEOUT,
                    f"等待响应超过 {timeout}s",
                    detail={"timeout_s": timeout},
                )
        return read_message(self._proc.stdout)

    def close(self) -> None:
        """关掉 stdin，服务端读到 EOF 会自行退出。"""
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()   # 赖着不走就强杀，端侧不能容忍僵尸进程
                self._proc.wait()

    # 支持 with 语句，保证异常时也能回收子进程
    def __enter__(self) -> SubprocessTransport:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
