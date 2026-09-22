from __future__ import annotations

import signal

import pytest

from faar.operations import check_termination, install_graceful_termination_handler


def test_termination_raises_system_exit_on_signal() -> None:
    install_graceful_termination_handler()
    try:
        signal.raise_signal(signal.SIGTERM)
        with pytest.raises(SystemExit) as exc_info:
            check_termination()
        assert exc_info.value.code == 128 + signal.SIGTERM
    finally:
        install_graceful_termination_handler()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
