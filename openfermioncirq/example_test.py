import pytest

from openfermioncirq import example


def test_example():
    assert example.make_circuit_diagram() == """
q: ───X───Z───
          │
r: ───────Z───
    """.strip()


def test_failure():
    with pytest.raises(ValueError):
        example.cause_failure()
