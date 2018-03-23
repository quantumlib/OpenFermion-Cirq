import cirq
import openfermion


def make_circuit_diagram():
    q = cirq.NamedQubit('q')
    r = cirq.NamedQubit('r')
    c = cirq.Circuit.from_ops(
        cirq.X(q),
        cirq.CZ(q, r)
    )
    return str(c)


def make_qubit_operator_str():
    a = openfermion.QubitOperator('X0 Z5')
    b = openfermion.QubitOperator('X5 Z7')
    return str(a * b)


def cause_failure():
    raise ValueError('failure')
