"""让 mock 服务端能作为独立进程运行：

    python -m miagent.mock_server

它从 stdin 读报文、往 stdout 写响应，日志走 stderr。
客户端用 SubprocessTransport 拉起它。
"""

from ..protocol import MiClawError, error_response
from ..transport import StdioServerTransport, log
from .server import MiClawMockServer


def main() -> None:
    """以 stdio 为传输启动 mock 服务端，逐行读报文、写响应，直到对端关闭。"""
    transport = StdioServerTransport()
    server = MiClawMockServer()
    log("[server] MiClaw mock 已启动，等待报文…")

    while True:
        try:
            raw = transport.receive()
        except MiClawError as e:
            # 报文连 JSON 都不是，拿不到 id，按 JSON-RPC 规定回 id=null
            transport.send(error_response(None, e).model_dump(exclude_none=True))
            continue

        if raw is None:                       # 对端关闭了管道
            log("[server] 对端已关闭，退出")
            break

        response = server.handle(raw)
        if response is not None:              # 通知类报文不回复
            transport.send(response)


if __name__ == "__main__":
    main()
