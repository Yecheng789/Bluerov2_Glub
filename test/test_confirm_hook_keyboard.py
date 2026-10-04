"""Tests for the dedicated fixed-hook confirmation keyboard client."""

from pathlib import Path
from types import SimpleNamespace

from bluerov2_control import confirm_hook_keyboard as module
import pytest


class FakeStream:
    """Small terminal-like stream used by unit tests."""

    def __init__(self, *, is_tty=True, reads=()):
        self._is_tty = is_tty
        self._reads = list(reads)

    def isatty(self):
        return self._is_tty

    def fileno(self):
        return 42

    def read(self, _size):
        return self._reads.pop(0)


class FakeFuture:
    """Controllable Future stand-in."""

    def __init__(self, *, done=False, result=None, error=None):
        self.is_done = done
        self.response = result
        self.error = error

    def done(self):
        return self.is_done

    def result(self):
        if self.error is not None:
            raise self.error
        return self.response


class FakeLogger:
    """Capture logger output without constructing an rclpy node."""

    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)


def test_confirmation_key_is_only_h_in_either_case():
    assert module.is_confirmation_key('h')
    assert module.is_confirmation_key('H')
    for key in ('', 'g', ' ', '\n', '\x1b'):
        assert not module.is_confirmation_key(key)


def test_read_key_nonblocking_handles_timeout_key_eof_and_ctrl_c():
    stream = FakeStream(reads=['H', '', '\x03'])

    assert module.read_key_nonblocking(
        stream,
        0.1,
        select_fn=lambda *_args: ([], [], []),
    ) is None
    assert module.read_key_nonblocking(
        stream,
        0.1,
        select_fn=lambda *_args: ([stream], [], []),
    ) == 'H'
    with pytest.raises(EOFError, match='stdin closed'):
        module.read_key_nonblocking(
            stream,
            0.1,
            select_fn=lambda *_args: ([stream], [], []),
        )
    with pytest.raises(KeyboardInterrupt):
        module.read_key_nonblocking(
            stream,
            0.1,
            select_fn=lambda *_args: ([stream], [], []),
        )


def test_cbreak_terminal_requires_a_real_tty():
    with pytest.raises(RuntimeError, match='stdin is not a TTY'):
        with module.CbreakTerminal(FakeStream(is_tty=False)):
            pytest.fail('non-TTY must never enter terminal mode')


def test_cbreak_terminal_restores_after_normal_exit_and_exception(
    monkeypatch,
):
    calls = []
    settings = ['saved-settings']
    monkeypatch.setattr(
        module.termios,
        'tcgetattr',
        lambda fd: calls.append(('get', fd)) or settings,
    )
    monkeypatch.setattr(
        module.tty,
        'setcbreak',
        lambda fd: calls.append(('cbreak', fd)),
    )
    monkeypatch.setattr(
        module.termios,
        'tcsetattr',
        lambda fd, when, value: calls.append(
            ('restore', fd, when, value)
        ),
    )
    monkeypatch.setattr(
        module.termios,
        'tcflush',
        lambda fd, queue: calls.append(('flush', fd, queue)),
    )

    with module.CbreakTerminal(FakeStream()) as terminal:
        terminal.discard_pending_input()
    assert calls[:3] == [
        ('get', 42),
        ('cbreak', 42),
        ('flush', 42, module.termios.TCIFLUSH),
    ]
    assert calls.count(
        ('restore', 42, module.termios.TCSADRAIN, settings)
    ) == 1
    assert calls.count(
        ('flush', 42, module.termios.TCIFLUSH)
    ) == 2

    with pytest.raises(ValueError, match='test failure'):
        with module.CbreakTerminal(FakeStream()):
            raise ValueError('test failure')
    assert calls.count(
        ('restore', 42, module.termios.TCSADRAIN, settings)
    ) == 2


def test_cbreak_terminal_attempts_restore_when_setup_fails(monkeypatch):
    restored = []
    monkeypatch.setattr(module.termios, 'tcgetattr', lambda _fd: ['saved'])
    monkeypatch.setattr(
        module.tty,
        'setcbreak',
        lambda _fd: (_ for _ in ()).throw(OSError('cbreak failed')),
    )
    monkeypatch.setattr(
        module.termios,
        'tcsetattr',
        lambda fd, when, value: restored.append((fd, when, value)),
    )

    with pytest.raises(OSError, match='cbreak failed'):
        with module.CbreakTerminal(FakeStream()):
            pass
    assert restored == [(42, module.termios.TCSADRAIN, ['saved'])]


def test_trigger_request_gate_blocks_duplicate_and_releases_on_result():
    future = FakeFuture(done=False, result=SimpleNamespace(success=True))
    requests = []
    client = SimpleNamespace(
        call_async=lambda request: requests.append(request) or future
    )
    gate = module.TriggerRequestGate(client)

    assert gate.begin() is future
    assert gate.begin() is None
    assert len(requests) == 1

    future.is_done = True
    assert gate.finish(future).success is True
    assert gate.pending_future is None


def test_trigger_request_gate_does_not_overwrite_completed_unread_request():
    first = FakeFuture(done=True, result=SimpleNamespace(success=True))
    second = FakeFuture(done=False)
    futures = iter([first, second])
    client = SimpleNamespace(call_async=lambda _request: next(futures))
    gate = module.TriggerRequestGate(client)

    assert gate.begin() is first
    assert gate.begin() is None
    assert gate.finish(first).success is True
    assert gate.begin() is second


def test_trigger_request_gate_releases_when_future_result_raises():
    future = FakeFuture(done=True, error=RuntimeError('service failed'))
    gate = module.TriggerRequestGate(
        SimpleNamespace(call_async=lambda _request: future)
    )
    gate.begin()

    with pytest.raises(RuntimeError, match='service failed'):
        gate.finish(future)
    assert gate.pending_future is None


def test_wait_for_future_spins_until_completion():
    future = FakeFuture(done=False)
    now = [10.0]
    spins = []

    def spin_once(_node, timeout_sec):
        spins.append(timeout_sec)
        now[0] += timeout_sec
        future.is_done = True

    module.wait_for_future(
        object(),
        future,
        1.0,
        spin_once_fn=spin_once,
        monotonic_fn=lambda: now[0],
    )
    assert spins == [pytest.approx(0.05)]


def test_wait_for_future_has_a_bounded_timeout():
    future = FakeFuture(done=False)
    now = [10.0]

    def spin_once(_node, timeout_sec):
        now[0] += timeout_sec

    with pytest.raises(TimeoutError, match='service response'):
        module.wait_for_future(
            object(),
            future,
            0.1,
            spin_once_fn=spin_once,
            monotonic_fn=lambda: now[0],
        )


class FakeTerminalContext:
    """Record context entry, exit, and input flushes."""

    instances = []

    def __init__(self, _stream):
        self.entered = False
        self.exited = False
        self.flush_count = 0
        self.__class__.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def discard_pending_input(self):
        self.flush_count += 1

    def __exit__(self, _exc_type, _exc, _traceback):
        self.exited = True
        return False


class FakeSessionNode:
    """Node-shaped object for operator-session tests."""

    service_name = module.DEFAULT_CONFIRM_HOOK_SERVICE
    service_wait_timeout_s = 2.0

    def __init__(self, responses, *, service_ready=True):
        self.responses = list(responses)
        self.service_ready = service_ready
        self.requests = 0
        self.logger = FakeLogger()

    def get_logger(self):
        return self.logger

    def wait_until_service_ready(self):
        return self.service_ready

    def request_confirmation(self):
        self.requests += 1
        return object()

    def response_for(self, _future):
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_operator_session_ignores_other_keys_and_exits_after_rejection(
    monkeypatch,
):
    keys = iter(['x', 'h', 'H'])
    node = FakeSessionNode([
        SimpleNamespace(success=False, message='not in WAIT_HOOK'),
    ])
    FakeTerminalContext.instances.clear()
    monkeypatch.setattr(module, 'CbreakTerminal', FakeTerminalContext)
    monkeypatch.setattr(
        module,
        'read_key_nonblocking',
        lambda *_args, **_kwargs: next(keys),
    )
    monkeypatch.setattr(module.rclpy, 'ok', lambda: True)

    assert module.run_operator_session(node, FakeStream()) == 2
    assert node.requests == 1
    assert len(node.logger.warnings) == 1
    assert 'rejected' in node.logger.warnings[0]
    terminal = FakeTerminalContext.instances[0]
    assert terminal.entered and terminal.exited
    assert terminal.flush_count == 1


def test_operator_session_exits_zero_after_one_accepted_request(monkeypatch):
    keys = iter(['H', 'h'])
    node = FakeSessionNode([
        SimpleNamespace(success=True, message='GO_BACK started'),
    ])
    FakeTerminalContext.instances.clear()
    monkeypatch.setattr(module, 'CbreakTerminal', FakeTerminalContext)
    monkeypatch.setattr(
        module,
        'read_key_nonblocking',
        lambda *_args, **_kwargs: next(keys),
    )
    monkeypatch.setattr(module.rclpy, 'ok', lambda: True)

    assert module.run_operator_session(node, FakeStream()) == 0
    assert node.requests == 1
    terminal = FakeTerminalContext.instances[0]
    assert terminal.entered and terminal.exited
    assert terminal.flush_count == 1


@pytest.mark.parametrize(
    'failure',
    [EOFError('closed'), RuntimeError('response failed')],
)
def test_operator_session_restores_terminal_on_eof_or_response_failure(
    monkeypatch,
    failure,
):
    node = FakeSessionNode(
        [failure] if isinstance(failure, RuntimeError) else []
    )
    FakeTerminalContext.instances.clear()
    monkeypatch.setattr(module, 'CbreakTerminal', FakeTerminalContext)
    monkeypatch.setattr(module.rclpy, 'ok', lambda: True)
    if isinstance(failure, EOFError):
        monkeypatch.setattr(
            module,
            'read_key_nonblocking',
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    else:
        monkeypatch.setattr(
            module,
            'read_key_nonblocking',
            lambda *_args, **_kwargs: 'h',
        )

    with pytest.raises(type(failure), match=str(failure)) as exc_info:
        module.run_operator_session(node, FakeStream())
    if isinstance(failure, RuntimeError):
        assert 'outcome may be unknown' in str(exc_info.value)
    terminal = FakeTerminalContext.instances[0]
    assert terminal.entered and terminal.exited


def test_operator_session_fails_clearly_before_terminal_if_service_missing(
    monkeypatch,
):
    node = FakeSessionNode([], service_ready=False)
    FakeTerminalContext.instances.clear()
    monkeypatch.setattr(module, 'CbreakTerminal', FakeTerminalContext)

    with pytest.raises(RuntimeError, match='service.*unavailable'):
        module.run_operator_session(node, FakeStream())
    assert FakeTerminalContext.instances == []


def test_operator_session_rejects_non_tty_before_service_wait():
    node = FakeSessionNode([])
    with pytest.raises(RuntimeError, match='stdin is not a TTY'):
        module.run_operator_session(node, FakeStream(is_tty=False))
    assert node.requests == 0


def test_main_rejects_non_tty_before_initializing_ros(monkeypatch, capsys):
    initialized = []
    monkeypatch.setattr(module.sys, 'stdin', FakeStream(is_tty=False))
    monkeypatch.setattr(
        module.rclpy,
        'init',
        lambda **_kwargs: initialized.append(True),
    )

    assert module.main([]) == 1
    assert initialized == []
    assert 'stdin is not a TTY' in capsys.readouterr().err


def test_package_installs_keyboard_client_and_std_srvs_dependency():
    package_root = Path(__file__).resolve().parents[1]
    setup_text = (package_root / 'setup.py').read_text(encoding='utf-8')
    manifest_text = (package_root / 'package.xml').read_text(
        encoding='utf-8'
    )

    assert 'confirm_hook_keyboard = ' in setup_text
    assert 'bluerov2_control.confirm_hook_keyboard:main' in setup_text
    assert '<exec_depend>std_srvs</exec_depend>' in manifest_text
