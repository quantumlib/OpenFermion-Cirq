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


from typing import Any

import cirq


def assert_equivalent_repr(
        value: Any,
        *,
        setup_code: str = ('import cirq\n'
                           'import numpy as np\n'
                           'import openfermioncirq as ofc\n'
                           'import openfermion as of\n')) -> None:
    """Checks that eval(repr(v)) == v.

    Args:
        value: A value whose repr should be evaluatable python
            code that produces an equivalent value.
        setup_code: Code that must be executed before the repr can be evaluated.
            Ideally this should just be a series of 'import' lines.
    """
    cirq.testing.assert_equivalent_repr(value, setup_code=setup_code)
