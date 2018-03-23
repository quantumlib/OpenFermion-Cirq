import pytest

from openfermioncirq import example


def test_circuit_diagram():
    assert example.make_circuit_diagram() == """
q: ───X───Z───
          │
r: ───────Z───
    """.strip()


def test_qubit_operator_str():
    assert example.make_qubit_operator_str() == '1j [X0 Y5 Z7]'


def test_failure():
    with pytest.raises(ValueError):
        example.cause_failure()
