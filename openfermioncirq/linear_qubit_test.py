# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from cirq.testing import EqualsTester

from openfermioncirq.linear_qubit import LinearQubit


def test_linear_qubit_init():
    q = LinearQubit(3)
    assert q.index == 3


def test_linear_qubit_eq():
    eq = EqualsTester()
    eq.make_equality_pair(lambda: LinearQubit(0))
    eq.make_equality_pair(lambda: LinearQubit(1))
    eq.make_equality_pair(lambda: LinearQubit(2))
    eq.make_equality_pair(lambda: LinearQubit(7))


def test_linear_qubit_is_adjacent():
    assert LinearQubit(0).is_adjacent(LinearQubit(1))
    assert LinearQubit(0).is_adjacent(LinearQubit(-1))

    assert not LinearQubit(0).is_adjacent(LinearQubit(2))
    assert not LinearQubit(0).is_adjacent(LinearQubit(-2))

    assert not LinearQubit(0).is_adjacent(LinearQubit(3))
    assert not LinearQubit(0).is_adjacent(LinearQubit(7))

    assert LinearQubit(500).is_adjacent(LinearQubit(501))
    assert not LinearQubit(500).is_adjacent(LinearQubit(5034))
