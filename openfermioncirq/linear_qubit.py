#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from cirq.ops import QubitId


class LinearQubit(QubitId):
    """A qubit on a line."""

    def __init__(self, index):
        self.index = index

    def is_adjacent(self, other: 'LinearQubit') -> bool:
        return abs(self.index - other.index) == 1

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.index == other.index

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((LinearQubit, self.index))

    def __repr__(self):
        return 'LinearQubit({})'.format(self.index)

    def __str__(self):
        return '{}'.format(self.index)
