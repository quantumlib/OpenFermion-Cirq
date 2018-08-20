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

from openfermioncirq.variational.letter_with_subscripts import (
        LetterWithSubscripts)


def test_letter_with_subscripts_init():
    symbol = LetterWithSubscripts('T', 0, 1)
    assert symbol.letter == 'T'
    assert symbol.subscripts == (0, 1)
    assert symbol.name == 'T_0_1'
