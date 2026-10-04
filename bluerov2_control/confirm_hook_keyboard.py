#!/usr/bin/env python3
"""
Confirm a successful hook with one operator key press.

This node is intentionally run in its own foreground terminal.  It keeps
terminal input out of the safety-critical MPC executor and translates an
operator's ``H`` key press into a synchronous ``std_srvs/Trigger`` request.
"""

from __future__ import annotations

import math
import select
import sys
import termios
import time
import tty
from typing import Any, Callable, Optional, TextIO

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


DEFAULT_CONFIRM_HOOK_SERVICE = '/bluerov2/fixed_hook/confirm_hook'


def is_confirmation_key(key: str) -> bool:
    """Return whether *key* is the operator hook-confirmation key."""
    return key in ('h', 'H')


def read_key_nonblocking(
    stream: TextIO,
    timeout_sec: float,
    select_fn: Callable[..., Any] = select.select,
) -> Optional[str]:
    """Read one character, returning ``None`` when no input is ready."""
    ready, _, _ = select_fn([stream], [], [], max(0.0, timeout_sec))
    if not ready:
        return None

    key = stream.read(1)
    if key == '':
        raise EOFError('stdin closed while waiting for hook confirmation')
    if key == '\x03':
        raise KeyboardInterrupt
    return key


class CbreakTerminal:
    """Temporarily put a real TTY into cbreak mode and always restore it."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.fd: Optional[int] = None
        self.original_settings: Any = None
        self.active = False

    def __enter__(self) -> 'CbreakTerminal':
        if not self.stream.isatty():
            raise RuntimeError(
                'stdin is not a TTY; run confirm_hook_keyboard in its own '
                'interactive foreground terminal'
            )

        self.fd = self.stream.fileno()
        self.original_settings = termios.tcgetattr(self.fd)
        try:
            tty.setcbreak(self.fd)
            # A key pressed before this dedicated session became ready must
            # never be replayed as a fresh Hook confirmation.
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except BaseException:
            self.restore()
            raise
        self.active = True
        return self

    def discard_pending_input(self) -> None:
        """Discard key-repeat bytes buffered while a request was in flight."""
        if self.fd is None:
            return
        try:
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except (OSError, termios.error):
            # Restoration is safety-critical; flushing convenience is not.
            pass

    def restore(self) -> None:
        """Restore the saved terminal settings; safe to call repeatedly."""
        if self.fd is None or self.original_settings is None:
            return
        try:
            termios.tcsetattr(
                self.fd,
                termios.TCSADRAIN,
                self.original_settings,
            )
        finally:
            self.active = False

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        self.restore()
        return False


class TriggerRequestGate:
    """Allow at most one asynchronous Trigger request at a time."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.pending_future: Any = None

    def begin(self) -> Any:
        """Start one request, or return ``None`` if one is already pending."""
        if self.pending_future is not None:
            return None
        self.pending_future = self.client.call_async(Trigger.Request())
        return self.pending_future

    def finish(self, future: Any) -> Any:
        """Return a completed response and release the in-flight gate."""
        if future is not self.pending_future:
            raise RuntimeError(
                'attempted to finish an unknown Trigger request'
            )
        if not future.done():
            raise RuntimeError(
                'attempted to finish an incomplete Trigger request'
            )
        try:
            return future.result()
        finally:
            self.pending_future = None


def wait_for_future(
    node: Node,
    future: Any,
    timeout_sec: float,
    *,
    spin_once_fn: Callable[..., Any] = rclpy.spin_once,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> None:
    """Spin until *future* completes, raising on a bounded timeout."""
    deadline = monotonic_fn() + timeout_sec
    while not future.done():
        remaining = deadline - monotonic_fn()
        if remaining <= 0.0:
            raise TimeoutError(
                'timed out waiting for the confirm-hook service response'
            )
        spin_once_fn(node, timeout_sec=min(0.05, remaining))


class ConfirmHookKeyboard(Node):
    """ROS client used by the one-shot operator keyboard program."""

    def __init__(self) -> None:
        super().__init__('confirm_hook_keyboard')
        self.declare_parameter(
            'service_name',
            DEFAULT_CONFIRM_HOOK_SERVICE,
        )
        self.declare_parameter('service_wait_timeout_s', 5.0)
        self.declare_parameter('response_timeout_s', 5.0)

        self.service_name = str(
            self.get_parameter('service_name').value
        ).strip()
        self.service_wait_timeout_s = self._positive_timeout_parameter(
            'service_wait_timeout_s'
        )
        self.response_timeout_s = self._positive_timeout_parameter(
            'response_timeout_s'
        )
        if not self.service_name:
            raise ValueError('service_name must not be empty')

        self.client = self.create_client(Trigger, self.service_name)
        self.request_gate = TriggerRequestGate(self.client)

    def _positive_timeout_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'{name} must be finite and > 0')
        return value

    def wait_until_service_ready(self) -> bool:
        """Wait a bounded time for the controller confirmation service."""
        return bool(
            self.client.wait_for_service(
                timeout_sec=self.service_wait_timeout_s
            )
        )

    def request_confirmation(self) -> Any:
        """Begin a confirmation request unless one is already in flight."""
        return self.request_gate.begin()

    def response_for(self, future: Any) -> Trigger.Response:
        """Wait for and return the response for *future*."""
        wait_for_future(self, future, self.response_timeout_s)
        response = self.request_gate.finish(future)
        if response is None:
            raise RuntimeError(
                'confirm-hook service completed without a response'
            )
        return response


def run_operator_session(
    node: ConfirmHookKeyboard,
    stream: Optional[TextIO] = None,
) -> int:
    """Run the interactive confirmation session until success or failure."""
    if stream is None:
        stream = sys.stdin
    if not stream.isatty():
        raise RuntimeError(
            'stdin is not a TTY; run confirm_hook_keyboard in its own '
            'interactive foreground terminal'
        )
    if not node.wait_until_service_ready():
        raise RuntimeError(
            f'confirm-hook service {node.service_name!r} was unavailable '
            f'after {node.service_wait_timeout_s:.1f}s; start the fixed-hook '
            'MPC launch before this keyboard tool'
        )

    node.get_logger().info(
        f'Connected to {node.service_name}. Wait for mission -> WAIT_HOOK; '
        'after visually confirming that the hook has succeeded, press H '
        'once to authorize GO_BACK. Ctrl-C cancels.'
    )
    with CbreakTerminal(stream) as terminal:
        while rclpy.ok():
            key = read_key_nonblocking(stream, timeout_sec=0.10)
            if key is None or not is_confirmation_key(key):
                continue

            future = node.request_confirmation()
            if future is None:
                node.get_logger().warning(
                    'A hook-confirmation request is already in flight; '
                    'duplicate H ignored.'
                )
                continue

            node.get_logger().info(
                'H pressed; waiting for the MPC controller to accept or '
                'reject the hook confirmation.'
            )
            try:
                try:
                    response = node.response_for(future)
                except Exception as exc:
                    raise RuntimeError(
                        'hook-confirmation request outcome may be unknown; '
                        'inspect the MPC mission state before retrying: '
                        f'{exc}'
                    ) from exc
            finally:
                terminal.discard_pending_input()

            if bool(response.success):
                detail = str(response.message).strip()
                suffix = f' Controller: {detail}' if detail else ''
                node.get_logger().info(
                    'Hook confirmation accepted; GO_BACK is authorized and '
                    'the keyboard tool will exit.'
                    + suffix
                )
                return 0

            detail = str(response.message).strip() or 'no reason provided'
            node.get_logger().warning(
                'Hook confirmation rejected by the controller: '
                f'{detail}. This one-shot keyboard tool will exit; verify '
                'WAIT_HOOK before starting it again.'
            )
            return 2

    raise RuntimeError('ROS shut down before hook confirmation was accepted')


def main(args=None) -> int:
    """Run the dedicated operator keyboard client."""
    if not sys.stdin.isatty():
        print(
            'confirm_hook_keyboard: stdin is not a TTY; run this tool in '
            'its own interactive foreground terminal',
            file=sys.stderr,
        )
        return 1

    node: Optional[ConfirmHookKeyboard] = None
    initialized = False
    try:
        rclpy.init(args=args)
        initialized = True
        node = ConfirmHookKeyboard()
        return run_operator_session(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().warning(
                'Keyboard tool cancelled by operator. If cancellation '
                'occurred after pressing H, inspect the MPC state before '
                'retrying because the request outcome may be unknown.'
            )
        return 130
    except Exception as exc:
        # This includes EOF, service, terminal, and response failures.
        if node is not None:
            node.get_logger().error(str(exc))
        else:
            print(f'confirm_hook_keyboard: {exc}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if initialized and rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
