"""传输层测试。不涉及任何协议语义，只验证「报文能原样送到对面」。"""

import io
import sys

import pytest

from miagent.protocol import ErrorCode, MiClawError
from miagent.transport import (
    StdioServerTransport,
    SubprocessTransport,
    read_message,
    write_message,
)


class TestFraming:
    def test_write_produces_exactly_one_line(self):
        buf = io.StringIO()
        write_message(buf, {"jsonrpc": "2.0", "id": 1, "method": "ping"})
        out = buf.getvalue()
        assert out.endswith("\n")
        assert out.count("\n") == 1

    def test_roundtrip(self):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"中文": "值"}}
        buf = io.StringIO()
        write_message(buf, msg)
        buf.seek(0)
        assert read_message(buf) == msg

    def test_newline_inside_content_does_not_break_framing(self):
        """正文里的换行必须被转义，否则会被 readline 切成两半。"""
        msg = {"id": 1, "text": "第一行\n第二行"}
        buf = io.StringIO()
        write_message(buf, msg)
        assert buf.getvalue().count("\n") == 1   # 只有分帧那一个换行
        buf.seek(0)
        assert read_message(buf) == msg

    def test_two_messages_read_back_in_order(self):
        buf = io.StringIO()
        write_message(buf, {"id": 1})
        write_message(buf, {"id": 2})
        buf.seek(0)
        assert read_message(buf)["id"] == 1
        assert read_message(buf)["id"] == 2

    def test_eof_returns_none(self):
        """EOF 返回 None，与「收到一条空消息」区分开。"""
        assert read_message(io.StringIO("")) is None

    def test_blank_lines_are_skipped(self):
        assert read_message(io.StringIO('\n\n{"id":1}\n'))["id"] == 1

    def test_malformed_json_becomes_mc_2003(self):
        with pytest.raises(MiClawError) as ei:
            read_message(io.StringIO("{不是合法 json}\n"))
        assert ei.value.code == ErrorCode.MC_INVALID_MESSAGE


class TestStdioServerTransport:
    def test_receive_and_send(self):
        transport = StdioServerTransport(
            stdin=io.StringIO('{"id":1,"method":"ping"}\n'),
            stdout=io.StringIO(),
        )
        assert transport.receive()["method"] == "ping"
        transport.send({"id": 1, "result": {}})
        assert transport._out.getvalue() == '{"id":1,"result":{}}\n'


class TestSubprocessTransport:
    """拉起一个真实子进程，验证跨进程收发。"""

    ECHO_SERVER = (
        "import sys, json\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if not line: continue\n"
        "    msg = json.loads(line)\n"
        "    msg['echoed'] = True\n"
        "    sys.stdout.write(json.dumps(msg) + '\\n'); sys.stdout.flush()\n"
    )

    def test_end_to_end_over_real_pipe(self):
        with SubprocessTransport([sys.executable, "-c", self.ECHO_SERVER]) as t:
            t.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
            resp = t.receive()
            assert resp["id"] == 1 and resp["echoed"] is True

    def test_multiple_messages_keep_order(self):
        with SubprocessTransport([sys.executable, "-c", self.ECHO_SERVER]) as t:
            for i in range(5):
                t.send({"id": i})
            for i in range(5):
                assert t.receive()["id"] == i

    def test_send_after_server_exit_raises_mc_1002(self):
        # 一个立刻退出的"服务端"
        with SubprocessTransport([sys.executable, "-c", "pass"]) as t:
            t._proc.wait()
            with pytest.raises(MiClawError) as ei:
                t.send({"id": 1})
            assert ei.value.code == ErrorCode.MC_TRANSPORT_CLOSED
