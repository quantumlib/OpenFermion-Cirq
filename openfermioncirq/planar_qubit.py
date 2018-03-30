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

from cirq.ops import QubitId


class PlanarQubit(QubitId):
    """A qubit on a 2-dimensional square lattice."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def is_adjacent(self, other: 'PlanarQubit') -> bool:
        return abs(self.x - other.x) + abs(self.y - other.y) == 1

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((PlanarQubit, self.x, self.y))

    def __repr__(self):
        return 'PlanarQubit({}, {})'.format(self.x, self.y)

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)
