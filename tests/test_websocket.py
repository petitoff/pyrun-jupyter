"""Tests for Jupyter WebSocket message collection."""

import json
from base64 import b64encode
from hashlib import sha256
from unittest.mock import Mock

import pytest
import websocket

from pyrun_jupyter.contents import FileTransferError, RemoteFileNotFoundError
from pyrun_jupyter.websocket import (
    BINARY_DOWNLOAD_TARGET,
    BINARY_TRANSFER_SETUP_CODE,
    BINARY_UPLOAD_TARGET,
    KernelWebSocket,
)


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


class FakeCommManager:
    """Stands in for ``ipykernel``'s comm manager while exercising kernel-side code."""

    def __init__(self):
        self.targets = {}

    def register_target(self, name, callback):
        self.targets[name] = callback


class FakeComm:
    """Minimal Comm double that records what the kernel-side handler sends."""

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

    def send(self, data, metadata=None, buffers=None):
        self.sent.append((data, buffers))

    def close(self):
        self.closed = True

    @property
    def payloads(self):
        return [data for data, _buffers in self.sent]


def _install_kernel_handlers():
    """Execute the setup code the client injects, and hand back its comm targets."""
    manager = FakeCommManager()
    shell = type("Shell", (), {"kernel": type("Kernel", (), {"comm_manager": manager})()})()
    namespace = {"get_ipython": lambda: shell}
    exec(BINARY_TRANSFER_SETUP_CODE, namespace)
    return manager


def _open_comm(manager, target, data):
    comm = FakeComm()
    manager.targets[target](comm, {"content": {"data": data}})
    return comm


def test_remote_binary_handler_verifies_and_atomically_installs_file(tmp_path):
    """The code installed in the kernel writes and verifies the streamed buffers."""
    manager = _install_kernel_handlers()

    destination = tmp_path / "models" / "weights.bin"
    content = b"\x00raw-binary-data\xff"
    comm = _open_comm(
        manager,
        BINARY_UPLOAD_TARGET,
        {"path": str(destination), "transfer_id": "test-transfer"},
    )

    assert comm.payloads == [{"status": "ready"}]
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
    assert comm.payloads[-1] == {
        "status": "complete",
        "size": len(content),
        "sha256": sha256(content).hexdigest(),
    }
    assert comm.closed
    assert not list(destination.parent.glob("*.pyrun-upload-*"))


def test_remote_download_handler_streams_raw_chunks(tmp_path):
    """The kernel-side download handler answers each pull with one raw buffer."""
    manager = _install_kernel_handlers()

    source = tmp_path / "outputs" / "model.bin"
    source.parent.mkdir(parents=True)
    content = b"\x00binary-checkpoint\xff"
    source.write_bytes(content)

    comm = _open_comm(
        manager,
        BINARY_DOWNLOAD_TARGET,
        {"path": str(source), "transfer_id": "test-transfer", "chunk_size": 8},
    )
    assert comm.payloads == [{"status": "ready", "size": len(content)}]

    collected = bytearray()
    sequence = 0
    while True:
        comm.message_callback({"content": {"data": {"action": "next", "sequence": sequence}}})
        data, buffers = comm.sent[-1]
        if data["status"] == "complete":
            break
        assert data["sequence"] == sequence
        assert len(buffers) == 1
        assert len(buffers[0]) <= 8
        collected += buffers[0]
        sequence += 1

    assert bytes(collected) == content
    assert comm.sent[-1][0] == {
        "status": "complete",
        "size": len(content),
        "sha256": sha256(content).hexdigest(),
    }
    assert comm.closed


def test_remote_download_handler_reports_missing_file(tmp_path):
    """A missing artifact is a distinguishable status, not a generic failure."""
    manager = _install_kernel_handlers()

    comm = _open_comm(
        manager,
        BINARY_DOWNLOAD_TARGET,
        {"path": str(tmp_path / "absent.bin"), "transfer_id": "t", "chunk_size": 8},
    )

    assert comm.payloads[-1]["status"] == "error"
    assert comm.payloads[-1]["code"] == "not_found"
    assert comm.closed


def test_download_file_verifies_and_atomically_installs(tmp_path):
    """Downloads land through a temporary file and are checked against the kernel digest."""
    content = b"checkpoint-bytes-0123456789"
    destination = tmp_path / "artifacts" / "model.bin"

    client = KernelWebSocket("ws://jupyter.example/api/kernels/id/channels")
    client.ws = Mock()
    client._binary_transfer_ready = True

    chunks = [content[i : i + 8] for i in range(0, len(content), 8)]
    replies = [({"status": "ready", "size": len(content)}, [])]
    replies += [
        ({"status": "chunk", "sequence": index, "size": len(chunk)}, [chunk])
        for index, chunk in enumerate(chunks)
    ]
    replies.append(
        ({"status": "complete", "size": len(content), "sha256": sha256(content).hexdigest()}, [])
    )
    client._wait_for_transfer_payload = Mock(side_effect=replies)

    written = client.download_file(
        "project/outputs/model.bin", destination, chunk_size=8, timeout=5.0
    )

    assert written == len(content)
    assert destination.read_bytes() == content
    assert not list(destination.parent.glob("*.pyrun-download-*"))

    text_messages = [json.loads(call.args[0]) for call in client.ws.send.call_args_list]
    assert text_messages[0]["header"]["msg_type"] == "comm_open"
    assert text_messages[0]["content"]["data"]["chunk_size"] == 8
    assert [message["content"]["data"]["sequence"] for message in text_messages[1:]] == list(
        range(len(chunks) + 1)
    )
    assert "base64" not in json.dumps(text_messages)


def test_download_file_rejects_a_corrupted_transfer(tmp_path):
    """A digest mismatch fails loudly and leaves no partial file behind."""
    destination = tmp_path / "model.bin"

    client = KernelWebSocket("ws://jupyter.example/api/kernels/id/channels")
    client.ws = Mock()
    client._binary_transfer_ready = True
    client._wait_for_transfer_payload = Mock(
        side_effect=[
            ({"status": "ready", "size": 4}, []),
            ({"status": "chunk", "sequence": 0, "size": 4}, [b"cccc"]),
            ({"status": "complete", "size": 4, "sha256": sha256(b"dddd").hexdigest()}, []),
        ]
    )

    with pytest.raises(FileTransferError, match="failed verification"):
        client.download_file("model.bin", destination, chunk_size=4, timeout=5.0)

    assert not destination.exists()
    assert not list(tmp_path.glob("*.pyrun-download-*"))


def test_download_file_propagates_missing_remote_file(tmp_path):
    """A kernel 'not_found' status surfaces as RemoteFileNotFoundError."""
    client = KernelWebSocket("ws://jupyter.example/api/kernels/id/channels")
    client.ws = Mock()
    client._binary_transfer_ready = True

    def raise_not_found(*_args, **_kwargs):
        raise RemoteFileNotFoundError("Remote file not found: absent.bin")

    client._wait_for_transfer_payload = Mock(side_effect=raise_not_found)

    with pytest.raises(RemoteFileNotFoundError):
        client.download_file("absent.bin", tmp_path / "absent.bin", timeout=5.0)

    assert not list(tmp_path.glob("*.pyrun-download-*"))


def test_transfer_error_statuses_are_classified(tmp_path):
    """Generic remote errors stay FileTransferError; not_found is narrowed."""
    client = KernelWebSocket("ws://jupyter.example/api/kernels/id/channels")
    client.ws = Mock()

    def reply(data):
        return json.dumps({"msg_type": "comm_msg", "content": {"comm_id": "comm", "data": data}})

    client.ws.recv.side_effect = [reply({"status": "error", "error": "disk full"})]
    with pytest.raises(FileTransferError, match="disk full") as generic:
        client._wait_for_transfer_payload("comm", "ready", timeout=1.0)
    assert not isinstance(generic.value, RemoteFileNotFoundError)

    client.ws.recv.side_effect = [
        reply({"status": "error", "code": "not_found", "error": "Remote file not found: x"})
    ]
    with pytest.raises(RemoteFileNotFoundError):
        client._wait_for_transfer_payload("comm", "ready", timeout=1.0)


class LoopbackComm:
    """Kernel-side Comm that serializes replies onto a client's receive queue."""

    def __init__(self, comm_id, outbox):
        self.comm_id = comm_id
        self.outbox = outbox
        self.message_callback = None
        self.closed = False

    def on_msg(self, callback):
        self.message_callback = callback
        return callback

    def on_close(self, callback):
        return callback

    def send(self, data, metadata=None, buffers=None):
        message = {
            "header": {"msg_type": "comm_msg"},
            "msg_type": "comm_msg",
            "content": {"comm_id": self.comm_id, "data": data},
        }
        if buffers:
            self.outbox.append(
                KernelWebSocket.serialize_binary_message(message, [bytes(b) for b in buffers])
            )
        else:
            self.outbox.append(json.dumps(message))

    def close(self):
        self.closed = True


class LoopbackSocket:
    """Routes real client frames into the real kernel handlers, in process."""

    def __init__(self, manager):
        self.manager = manager
        self.outbox = []
        self.comms = {}
        self.frames = []

    def settimeout(self, _timeout):
        pass

    def send(self, payload):
        self.frames.append(payload)
        self._dispatch(json.loads(payload))

    def send_binary(self, payload):
        self.frames.append(payload)
        self._dispatch(KernelWebSocket.deserialize_binary_message(payload))

    def recv(self):
        if not self.outbox:
            raise websocket.WebSocketTimeoutException()
        frame = self.outbox.pop(0)
        self.frames.append(frame)
        return frame

    def _dispatch(self, message):
        content = message["content"]
        msg_type = message.get("msg_type") or message["header"]["msg_type"]
        if msg_type == "comm_open":
            comm = LoopbackComm(content["comm_id"], self.outbox)
            self.comms[content["comm_id"]] = comm
            self.manager.targets[content["target_name"]](
                comm, {"content": {"data": content["data"]}}
            )
        elif msg_type == "comm_msg":
            comm = self.comms[content["comm_id"]]
            comm.message_callback(
                {
                    "content": {"data": content["data"]},
                    "buffers": message.get("buffers", []),
                }
            )


def _loopback_client():
    client = KernelWebSocket("ws://jupyter.example/api/kernels/id/channels")
    client.ws = LoopbackSocket(_install_kernel_handlers())
    client._binary_transfer_ready = True
    return client


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\x00\xff" * 5000,
        bytes(range(256)) * 40,
    ],
    ids=["empty", "repeating-binary", "full-byte-range"],
)
def test_binary_round_trip_between_client_and_kernel(tmp_path, content):
    """Upload then download through the real protocol must return the exact bytes."""
    client = _loopback_client()
    source = tmp_path / "source.bin"
    source.write_bytes(content)

    assert client.upload_file(source, str(tmp_path / "remote.bin"), chunk_size=1024, timeout=5.0)
    assert (tmp_path / "remote.bin").read_bytes() == content

    destination = tmp_path / "returned" / "model.bin"
    written = client.download_file(
        str(tmp_path / "remote.bin"), destination, chunk_size=1024, timeout=5.0
    )

    assert written == len(content)
    assert destination.read_bytes() == content
    assert not list(destination.parent.glob("*.pyrun-download-*"))


def test_round_trip_never_puts_base64_on_the_wire(tmp_path):
    """Payload bytes travel as raw buffers, not encoded into any text frame."""
    client = _loopback_client()
    content = b"\x00\x01\x02sentinel-payload\xfd\xfe\xff"
    source = tmp_path / "source.bin"
    source.write_bytes(content)

    client.upload_file(source, str(tmp_path / "remote.bin"), chunk_size=8, timeout=5.0)
    client.download_file(str(tmp_path / "remote.bin"), tmp_path / "back.bin", 8, timeout=5.0)

    text_frames = [frame for frame in client.ws.frames if isinstance(frame, str)]
    assert text_frames, "expected JSON control frames on the wire"
    joined = "".join(text_frames)
    assert "base64" not in joined
    assert b64encode(content).decode("ascii") not in joined

    # The payload only ever appears inside binary frames, as raw bytes.
    binary_frames = [frame for frame in client.ws.frames if isinstance(frame, bytes)]
    assert any(content[:8] in frame for frame in binary_frames)
