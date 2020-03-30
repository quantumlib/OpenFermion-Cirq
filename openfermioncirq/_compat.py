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
from typing import Any, Callable
import functools

import pytest


def deprecated_test(test: Callable) -> Callable:
    """Marks a test as using deprecated functionality.

    Ensures the test is executed within the `pytest.deprecated_call()` context.

    Args:
        test: The test.

    Returns:
        The decorated test.
    """

    @functools.wraps(test)
    def decorated_test(*args, **kwargs) -> Any:
        with pytest.deprecated_call():
            test(*args, **kwargs)

    return decorated_test
