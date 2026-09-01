"""Tests for Jupyter WebSocket message collection."""

import json
from unittest.mock import Mock

import websocket

from pyrun_jupyter.websocket import KernelWebSocket


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
