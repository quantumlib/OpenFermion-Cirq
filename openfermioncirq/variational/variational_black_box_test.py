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

import pytest

from openfermioncirq.testing import ExampleAnsatz, ExampleVariationalObjective
from openfermioncirq.variational.variational_black_box import (
        VariationalBlackBox)


def test_variational_black_box_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = VariationalBlackBox(ExampleAnsatz(), ExampleVariationalObjective())


def test_variational_black_box_is_abstract_must_implement():
    class Missing(VariationalBlackBox):
        pass

    with pytest.raises(TypeError):
        _ = Missing(ExampleAnsatz(), ExampleVariationalObjective())


def test_variational_black_box_is_abstract_can_implement():
    class Included(VariationalBlackBox):
        def evaluate_noiseless(self, x):
            pass

    assert isinstance(Included(ExampleAnsatz(), ExampleVariationalObjective()),
                      VariationalBlackBox)
