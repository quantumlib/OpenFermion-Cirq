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

from openfermioncirq.planar_qubit import PlanarQubit


def test_planar_qubit_init():
    q = PlanarQubit(3, 4)
    assert q.x == 3
    assert q.y == 4


def test_planar_qubit_eq():
    eq = EqualsTester()
    eq.make_equality_pair(lambda: PlanarQubit(0, 0))
    eq.make_equality_pair(lambda: PlanarQubit(1, 0))
    eq.make_equality_pair(lambda: PlanarQubit(0, 1))
    eq.make_equality_pair(lambda: PlanarQubit(50, 25))


def test_planar_qubit_is_adjacent():
    assert PlanarQubit(0, 0).is_adjacent(PlanarQubit(0, 1))
    assert PlanarQubit(0, 0).is_adjacent(PlanarQubit(0, -1))
    assert PlanarQubit(0, 0).is_adjacent(PlanarQubit(1, 0))
    assert PlanarQubit(0, 0).is_adjacent(PlanarQubit(-1, 0))

    assert not PlanarQubit(0, 0).is_adjacent(PlanarQubit(+1, -1))
    assert not PlanarQubit(0, 0).is_adjacent(PlanarQubit(+1, +1))
    assert not PlanarQubit(0, 0).is_adjacent(PlanarQubit(-1, -1))
    assert not PlanarQubit(0, 0).is_adjacent(PlanarQubit(-1, +1))

    assert not PlanarQubit(0, 0).is_adjacent(PlanarQubit(2, 0))

    assert PlanarQubit(500, 999).is_adjacent(PlanarQubit(501, 999))
    assert not PlanarQubit(500, 999).is_adjacent(PlanarQubit(5034, 999))
