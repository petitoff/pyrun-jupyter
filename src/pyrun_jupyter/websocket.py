"""WebSocket handler for Jupyter kernel communication."""

import json
import struct
import time
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

import websocket

from .contents import FileTransferError
from .result import ExecutionResult

BINARY_TRANSFER_TARGET = "pyrun_jupyter.binary_file_transfer.v1"
BINARY_TRANSFER_READY_MARKER = "__PYRUN_BINARY_TRANSFER_READY__"
DEFAULT_BINARY_CHUNK_SIZE = 4 * 1024 * 1024

BINARY_TRANSFER_SETUP_CODE = f"""\
import hashlib as _pyrun_hashlib
import os as _pyrun_os

def _pyrun_binary_transfer_target(comm, open_msg):
    data = open_msg.get("content", {{}}).get("data", {{}})
    remote_path = data.get("path")
    if not isinstance(remote_path, str) or not remote_path:
        comm.send({{"status": "error", "error": "A non-empty destination path is required"}})
        comm.close()
        return

    project_root = globals().get("_PYRUN_JUPYTER_ROOT")
    if project_root is None:
        project_root = _pyrun_os.getcwd()
        globals()["_PYRUN_JUPYTER_ROOT"] = project_root
    if not _pyrun_os.path.isabs(remote_path):
        remote_path = _pyrun_os.path.abspath(_pyrun_os.path.join(project_root, remote_path))

    remote_parent = _pyrun_os.path.dirname(remote_path)
    if remote_parent:
        _pyrun_os.makedirs(remote_parent, exist_ok=True)
    temporary_path = remote_path + ".pyrun-upload-" + data.get("transfer_id", "unknown")
    state = {{
        "file": None,
        "temporary_path": temporary_path,
        "remote_path": remote_path,
        "size": 0,
        "sequence": 0,
        "sha256": _pyrun_hashlib.sha256(),
        "complete": False,
    }}

    def cleanup():
        file_object = state.get("file")
        if file_object is not None and not file_object.closed:
            file_object.close()
        if not state["complete"] and _pyrun_os.path.exists(temporary_path):
            _pyrun_os.unlink(temporary_path)

    try:
        state["file"] = open(temporary_path, "wb")
    except Exception as exc:
        comm.send({{"status": "error", "error": str(exc)}})
        comm.close()
        return

    @comm.on_msg
    def receive(message):
        payload = message.get("content", {{}}).get("data", {{}})
        action = payload.get("action")
        try:
            if action == "chunk":
                buffers = message.get("buffers", [])
                sequence = payload.get("sequence")
                if sequence != state["sequence"]:
                    raise ValueError("Unexpected chunk sequence")
                if len(buffers) != 1:
                    raise ValueError("Each chunk message must contain one binary buffer")
                chunk = buffers[0]
                state["file"].write(chunk)
                state["sha256"].update(chunk)
                state["size"] += len(chunk)
                state["sequence"] += 1
                comm.send({{
                    "status": "ack",
                    "sequence": sequence,
                    "written": state["size"],
                }})
            elif action == "finish":
                state["file"].close()
                expected_size = payload.get("size")
                expected_digest = payload.get("sha256")
                actual_digest = state["sha256"].hexdigest()
                if state["size"] != expected_size:
                    raise ValueError("Transferred file size does not match")
                if actual_digest != expected_digest:
                    raise ValueError("Transferred file checksum does not match")
                _pyrun_os.replace(temporary_path, remote_path)
                state["complete"] = True
                comm.send({{
                    "status": "complete",
                    "size": state["size"],
                    "sha256": actual_digest,
                }})
                comm.close()
            elif action == "abort":
                cleanup()
                comm.close()
            else:
                raise ValueError("Unknown binary transfer action")
        except Exception as exc:
            cleanup()
            comm.send({{"status": "error", "error": str(exc)}})
            comm.close()

    @comm.on_close
    def close(_message):
        cleanup()

    comm.send({{"status": "ready"}})

get_ipython().kernel.comm_manager.register_target(
    {BINARY_TRANSFER_TARGET!r},
    _pyrun_binary_transfer_target,
)
print({BINARY_TRANSFER_READY_MARKER!r})
"""


class KernelWebSocket:
    """WebSocket connection to a Jupyter kernel for code execution.

    Handles the Jupyter messaging protocol over WebSocket for:
    - Sending execute_request messages
    - Receiving stream (stdout/stderr), execute_result, and error messages
    - Collecting all outputs into ExecutionResult
    """

    def __init__(self, ws_url: str, headers: Optional[Dict[str, str]] = None) -> None:
        """Initialize WebSocket connection.

        Args:
            ws_url: WebSocket URL for the kernel (ws://server/api/kernels/{id}/channels)
            headers: Optional headers including auth token
        """
        self.ws_url = ws_url
        self.headers = headers or {}
        self.ws: Optional[websocket.WebSocket] = None
        self._msg_id: Optional[str] = None
        self._session_id = str(uuid.uuid4())
        self._binary_transfer_ready = False

    def connect(self) -> None:
        """Establish WebSocket connection."""
        self.ws = websocket.create_connection(
            self.ws_url, header=[f"{k}: {v}" for k, v in self.headers.items()], timeout=30
        )

    def close(self) -> None:
        """Close WebSocket connection."""
        if self.ws:
            self.ws.close()
            self.ws = None
        self._binary_transfer_ready = False

    @staticmethod
    def serialize_binary_message(message: Dict[str, Any], buffers: List[bytes]) -> bytes:
        """Serialize raw buffers using Jupyter's default binary WebSocket framing."""
        message_bytes = json.dumps(message).encode("utf-8")
        parts = [message_bytes, *buffers]
        offsets = [4 * (len(parts) + 1)]
        for part in parts[:-1]:
            offsets.append(offsets[-1] + len(part))
        header = struct.pack("!" + "I" * (len(parts) + 1), len(parts), *offsets)
        return b"".join([header, *parts])

    @staticmethod
    def deserialize_binary_message(message: bytes) -> Dict[str, Any]:
        """Deserialize the default Jupyter binary WebSocket framing."""
        if len(message) < 8:
            raise ValueError("Invalid binary Jupyter message")
        part_count = struct.unpack("!I", message[:4])[0]
        header_size = 4 * (part_count + 1)
        if part_count < 1 or len(message) < header_size:
            raise ValueError("Invalid binary Jupyter message header")
        offsets = list(struct.unpack("!" + "I" * part_count, message[4:header_size]))
        if offsets[0] != header_size or any(
            start > stop or start < header_size or stop > len(message)
            for start, stop in zip(offsets, offsets[1:])
        ):
            raise ValueError("Invalid binary Jupyter message offsets")
        offsets.append(len(message))
        parts = [message[start:stop] for start, stop in zip(offsets, offsets[1:])]
        decoded = json.loads(parts[0].decode("utf-8"))
        decoded["buffers"] = parts[1:]
        return decoded

    def _decode_message(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, bytes):
            return self.deserialize_binary_message(response)
        return json.loads(response)

    def _create_message(
        self, msg_type: str, content: Dict[str, Any], channel: str = "shell"
    ) -> Dict[str, Any]:
        self._msg_id = str(uuid.uuid4())
        return {
            "header": {
                "msg_id": self._msg_id,
                "msg_type": msg_type,
                "username": "pyrun_jupyter",
                "session": self._session_id,
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": content,
            "channel": channel,
        }

    def _send_message(
        self, msg_type: str, content: Dict[str, Any], buffers: Optional[List[bytes]] = None
    ) -> None:
        if not self.ws:
            self.connect()
        if not self.ws:
            raise FileTransferError("WebSocket connection was not established")
        message = self._create_message(msg_type, content)
        if buffers:
            self.ws.send_binary(self.serialize_binary_message(message, buffers))
        else:
            self.ws.send(json.dumps(message))

    def _wait_for_transfer_status(
        self, comm_id: str, expected_status: str, timeout: float
    ) -> Dict[str, Any]:
        if not self.ws:
            raise FileTransferError("WebSocket connection was not established")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FileTransferError(f"Binary transfer timed out waiting for {expected_status}")
            self.ws.settimeout(remaining)
            try:
                reply = self._decode_message(self.ws.recv())
            except websocket.WebSocketTimeoutException as exc:
                raise FileTransferError(
                    f"Binary transfer timed out waiting for {expected_status}"
                ) from exc
            msg_type = reply.get("msg_type") or reply.get("header", {}).get("msg_type")
            content = reply.get("content", {})
            if content.get("comm_id") != comm_id:
                continue
            if msg_type == "comm_close":
                raise FileTransferError("Kernel closed the binary transfer unexpectedly")
            if msg_type != "comm_msg":
                continue
            data = content.get("data", {})
            if data.get("status") == "error":
                raise FileTransferError(data.get("error", "Remote binary transfer failed"))
            if data.get("status") == expected_status:
                return data

    def _ensure_binary_transfer(self, timeout: float) -> None:
        if self._binary_transfer_ready:
            return
        result = self.execute(BINARY_TRANSFER_SETUP_CODE, timeout=timeout)
        if result.has_error or BINARY_TRANSFER_READY_MARKER not in result.stdout:
            raise FileTransferError(
                result.error or "Failed to register the binary transfer target in the kernel"
            )
        self._binary_transfer_ready = True

    def upload_file(
        self,
        local_path: Path,
        remote_path: str,
        chunk_size: int = DEFAULT_BINARY_CHUNK_SIZE,
        timeout: float = 120.0,
    ) -> bool:
        """Stream a file to the kernel as binary WebSocket buffers."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not local_path.is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        if not remote_path:
            raise ValueError("remote_path must not be empty")
        self._ensure_binary_transfer(timeout)

        comm_id = str(uuid.uuid4())
        self._send_message(
            "comm_open",
            {
                "comm_id": comm_id,
                "target_name": BINARY_TRANSFER_TARGET,
                "data": {"path": remote_path, "transfer_id": comm_id},
            },
        )
        try:
            self._wait_for_transfer_status(comm_id, "ready", timeout)

            digest = sha256()
            transferred = 0
            sequence = 0
            with local_path.open("rb") as source:
                while True:
                    chunk = source.read(chunk_size)
                    if not chunk:
                        break
                    self._send_message(
                        "comm_msg",
                        {
                            "comm_id": comm_id,
                            "data": {"action": "chunk", "sequence": sequence},
                        },
                        buffers=[chunk],
                    )
                    acknowledgement = self._wait_for_transfer_status(comm_id, "ack", timeout)
                    if acknowledgement.get("sequence") != sequence:
                        raise FileTransferError("Kernel acknowledged an unexpected chunk")
                    digest.update(chunk)
                    transferred += len(chunk)
                    sequence += 1

            self._send_message(
                "comm_msg",
                {
                    "comm_id": comm_id,
                    "data": {
                        "action": "finish",
                        "size": transferred,
                        "sha256": digest.hexdigest(),
                    },
                },
            )
            completed = self._wait_for_transfer_status(comm_id, "complete", timeout)
            if (
                completed.get("size") != transferred
                or completed.get("sha256") != digest.hexdigest()
            ):
                raise FileTransferError("Kernel returned invalid transfer confirmation")
            return True
        except Exception:
            try:
                self._send_message(
                    "comm_msg",
                    {"comm_id": comm_id, "data": {"action": "abort"}},
                )
            except Exception:
                pass
            raise

    def execute(self, code: str, timeout: float = 60.0) -> ExecutionResult:
        """Execute code and collect results.

        Args:
            code: Python code to execute
            timeout: Maximum time to wait for execution (seconds)

        Returns:
            ExecutionResult with stdout, stderr, and any errors
        """
        if not self.ws:
            self.connect()
        ws = self.ws
        if ws is None:
            raise RuntimeError("WebSocket connection was not established")

        # Create execute_request message
        self._msg_id = str(uuid.uuid4())
        msg = self._create_execute_request(code)

        # Send message
        ws.send(json.dumps(msg))

        # Collect responses
        result = ExecutionResult()
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []

        deadline = time.monotonic() + timeout
        received_reply = False
        received_idle = False

        while not (received_reply and received_idle):
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise websocket.WebSocketTimeoutException()
                ws.settimeout(remaining)
                response = ws.recv()
                reply = self._decode_message(response)

                # Only process messages for our request
                parent_msg_id = reply.get("parent_header", {}).get("msg_id")
                if parent_msg_id != self._msg_id:
                    continue

                msg_type = reply.get("msg_type", "")
                content = reply.get("content", {})

                if msg_type == "stream":
                    # stdout or stderr output
                    name = content.get("name", "stdout")
                    text = content.get("text", "")
                    if name == "stdout":
                        stdout_parts.append(text)
                    else:
                        stderr_parts.append(text)

                elif msg_type == "execute_result":
                    # Rich output (e.g., display of variables)
                    result.data = content.get("data", {})
                    result.execution_count = content.get("execution_count")

                elif msg_type == "display_data":
                    # Additional display outputs
                    result.display_data.append(content.get("data", {}))

                elif msg_type == "error":
                    # Execution error
                    result.success = False
                    result.error_name = content.get("ename", "Error")
                    result.error = content.get("evalue", "")
                    result.error_traceback = content.get("traceback", [])

                elif msg_type == "execute_reply":
                    # The shell reply can arrive before the final IOPub output.
                    status = content.get("status")
                    if status == "error":
                        result.success = False
                        if not result.error:
                            result.error_name = content.get("ename", "Error")
                            result.error = content.get("evalue", "")
                    result.execution_count = content.get("execution_count")
                    received_reply = True

                elif msg_type == "status":
                    # Kernel status updates (busy/idle)
                    execution_state = content.get("execution_state")
                    if execution_state == "idle":
                        received_idle = True

            except websocket.WebSocketTimeoutException:
                result.success = False
                result.error = "Execution timed out"
                result.error_name = "TimeoutError"
                break

        result.stdout = "".join(stdout_parts)
        result.stderr = "".join(stderr_parts)

        return result

    def _create_execute_request(self, code: str) -> Dict[str, Any]:
        """Create a Jupyter execute_request message.

        Args:
            code: Python code to execute

        Returns:
            Message dictionary following Jupyter protocol
        """
        return self._create_message(
            "execute_request",
            {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
        )
