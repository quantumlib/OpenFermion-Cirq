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

import sympy

import cirq
import openfermioncirq as ofc

from openfermioncirq.variational.letter_with_subscripts import (
        LetterWithSubscripts)


def test_letter_with_subscripts_init():
    symbol = LetterWithSubscripts('T', 0, 1)
    assert symbol.letter == 'T'
    assert symbol.subscripts == (0, 1)
    assert str(symbol) == 'T_0_1'


def test_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(LetterWithSubscripts('T', 0, 1),
                          LetterWithSubscripts('T', 0, 1),
                          sympy.Symbol('T_0_1'))
    eq.add_equality_group(LetterWithSubscripts('S', 0))
    eq.add_equality_group(LetterWithSubscripts('T', 0, 2))


def test_substitute_works():
    assert LetterWithSubscripts('T', 1, 2).subs({'T_1_2': 5}) == 5


def test_repr():
    ofc.testing.assert_equivalent_repr(LetterWithSubscripts('T', 1, 2))
