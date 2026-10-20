"""Length-prefixed msgpack RPC between the host and the simulator process.

Shared by both sides: the host (any Python >= 3.10) and the simulator
(RoboDojo's Isaac Python). No pickle, no retries: a lost reply is an error,
because a physical action must never be executed twice.
"""
import socket
import struct
import uuid
import zlib

import msgpack
import numpy as np

VERSION = 1
MAX_BYTES = 128 * 1024 * 1024


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'buif':
            raise TypeError(f'unsupported array dtype: {value.dtype}')
        value = np.ascontiguousarray(value)
        return msgpack.ExtType(42, msgpack.packb((value.dtype.str, value.shape, value.tobytes()), use_bin_type=True))
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'unsupported RPC value: {type(value).__name__}')


def _decode(code, payload):
    if code != 42:
        raise ValueError('unknown RPC extension')
    dtype, shape, data = msgpack.unpackb(payload, raw=False)
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()


def _read_exact(sock, length):
    data = bytearray()
    while len(data) < length:
        block = sock.recv(min(length - len(data), 1 << 20))
        if not block:
            raise EOFError('RPC connection closed')
        data.extend(block)
    return bytes(data)


def send_packet(sock, value):
    payload = zlib.compress(msgpack.packb(value, default=_encode, use_bin_type=True), level=1)
    if len(payload) > MAX_BYTES:
        raise ValueError('RPC message too large')
    sock.sendall(struct.pack('!I', len(payload)) + payload)


def receive_packet(sock):
    length, = struct.unpack('!I', _read_exact(sock, 4))
    if not 0 < length <= MAX_BYTES:
        raise ValueError('invalid RPC packet length')
    return msgpack.unpackb(zlib.decompress(_read_exact(sock, length)), raw=False, ext_hook=_decode)


class RPCClient:
    def __init__(self, port, host='127.0.0.1', timeout=900):
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def request(self, op, **args):
        if self.sock is None:
            raise RuntimeError('RPC client closed; the episode cannot continue')
        request_id = uuid.uuid4().hex
        try:
            send_packet(self.sock, dict(version=VERSION, request_id=request_id, op=op, args=args))
            reply = receive_packet(self.sock)
        except Exception:
            self.close()
            raise
        if reply.get('request_id') != request_id:
            self.close()
            raise RuntimeError('RPC reply identity mismatch')
        if not reply.get('ok'):
            raise RuntimeError(reply.get('error', 'RPC request failed'))
        return reply['result']

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
