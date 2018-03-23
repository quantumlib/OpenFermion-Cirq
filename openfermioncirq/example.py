import cirq


def make_circuit_diagram():
    q = cirq.NamedQubit('q')
    r = cirq.NamedQubit('r')
    c = cirq.Circuit.from_ops(
        cirq.X(q),
        cirq.CZ(q, r)
    )
    return str(c)


def cause_failure():
    raise ValueError('failure')
