"""Tests for Jupyter WebSocket message collection."""

import json
from hashlib import sha256
from unittest.mock import Mock

import websocket

from pyrun_jupyter.websocket import BINARY_TRANSFER_SETUP_CODE, KernelWebSocket


def _message(client, msg_type, content):
    return json.dumps(
        {
            "parent_header": {"msg_id": client._msg_id},
            "msg_type": msg_type,
            "content": content,
        }
    )


def test_execute_waits_for_reply_and_idle_before_returning():
    """IOPub output arriving after the shell reply must not be dropped."""
    client = KernelWebSocket("ws://jupyter.example/api/kernels/id/channels")
    client.ws = Mock()
    responses = [
        ("execute_reply", {"status": "ok", "execution_count": 1}),
        ("stream", {"name": "stdout", "text": "complete output\n"}),
        ("status", {"execution_state": "idle"}),
    ]
    client.ws.recv.side_effect = lambda: _message(client, *responses.pop(0))

    result = client.execute("print('complete output')")

    assert result.success
    assert result.stdout == "complete output\n"
    assert result.execution_count == 1


def test_execute_returns_timeout_result():
    """Socket timeouts are represented as failed execution results."""
    client = KernelWebSocket("ws://jupyter.example/api/kernels/id/channels")
    client.ws = Mock()
    client.ws.recv.side_effect = websocket.WebSocketTimeoutException()

    result = client.execute("while True: pass", timeout=0.01)

    assert result.has_error
    assert result.error_name == "TimeoutError"


def test_binary_message_round_trip_preserves_raw_buffers():
    """Default Jupyter binary framing must preserve buffers byte-for-byte."""
    message = {
        "header": {"msg_id": "id", "msg_type": "comm_msg"},
        "parent_header": {},
        "metadata": {},
        "content": {"comm_id": "comm", "data": {"action": "chunk"}},
        "channel": "shell",
    }
    buffers = [b"\x00\x01binary", b"second-buffer"]

    encoded = KernelWebSocket.serialize_binary_message(message, buffers)
    decoded = KernelWebSocket.deserialize_binary_message(encoded)

    assert decoded["content"] == message["content"]
    assert decoded["buffers"] == buffers


def test_upload_file_streams_binary_chunks(tmp_path):
    """Uploads use bounded raw buffers instead of embedding base64 in source code."""
    content = b"abcdefghij"
    source = tmp_path / "model.bin"
    source.write_bytes(content)

    client = KernelWebSocket("ws://jupyter.example/api/kernels/id/channels")
    client.ws = Mock()
    client._binary_transfer_ready = True
    acknowledgement_sequence = iter(range(3))

    def transfer_status(_comm_id, expected_status, _timeout):
        if expected_status == "ready":
            return {"status": "ready"}
        if expected_status == "ack":
            return {"status": "ack", "sequence": next(acknowledgement_sequence)}
        return {
            "status": "complete",
            "size": len(content),
            "sha256": sha256(content).hexdigest(),
        }

    client._wait_for_transfer_status = Mock(side_effect=transfer_status)

    assert client.upload_file(source, "remote/model.bin", chunk_size=4)

    binary_messages = [
        KernelWebSocket.deserialize_binary_message(call.args[0])
        for call in client.ws.send_binary.call_args_list
    ]
    assert len(binary_messages) == 3
    assert b"".join(message["buffers"][0] for message in binary_messages) == content
    assert all(message["content"]["data"]["action"] == "chunk" for message in binary_messages)

    text_messages = [json.loads(call.args[0]) for call in client.ws.send.call_args_list]
    assert text_messages[0]["header"]["msg_type"] == "comm_open"
    assert text_messages[-1]["content"]["data"]["action"] == "finish"
    assert "base64" not in json.dumps(text_messages)


def test_remote_binary_handler_verifies_and_atomically_installs_file(tmp_path):
    """The code installed in the kernel writes and verifies the streamed buffers."""

    class FakeCommManager:
        def register_target(self, name, callback):
            self.name = name
            self.callback = callback

    class FakeComm:
        def __init__(self):
            self.sent = []
            self.message_callback = None
            self.close_callback = None
            self.closed = False

        def on_msg(self, callback):
            self.message_callback = callback
            return callback

        def on_close(self, callback):
            self.close_callback = callback
            return callback

        def send(self, data):
            self.sent.append(data)

        def close(self):
            self.closed = True

    manager = FakeCommManager()
    shell = type("Shell", (), {"kernel": type("Kernel", (), {"comm_manager": manager})()})()
    namespace = {"get_ipython": lambda: shell}
    exec(BINARY_TRANSFER_SETUP_CODE, namespace)

    destination = tmp_path / "models" / "weights.bin"
    content = b"\x00raw-binary-data\xff"
    comm = FakeComm()
    manager.callback(
        comm,
        {
            "content": {
                "data": {
                    "path": str(destination),
                    "transfer_id": "test-transfer",
                }
            }
        },
    )

    assert comm.sent == [{"status": "ready"}]
    comm.message_callback(
        {
            "content": {"data": {"action": "chunk", "sequence": 0}},
            "buffers": [memoryview(content)],
        }
    )
    comm.message_callback(
        {
            "content": {
                "data": {
                    "action": "finish",
                    "size": len(content),
                    "sha256": sha256(content).hexdigest(),
                }
            }
        }
    )

    assert destination.read_bytes() == content
    assert comm.sent[-1] == {
        "status": "complete",
        "size": len(content),
        "sha256": sha256(content).hexdigest(),
    }
    assert comm.closed
    assert not list(destination.parent.glob("*.pyrun-upload-*"))
