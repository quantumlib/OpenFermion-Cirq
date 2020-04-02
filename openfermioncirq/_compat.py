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
from typing import Any, Callable, Dict, Tuple
from types import ModuleType
import functools
import warnings

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


def wrap_module(module: ModuleType,
                deprecated_attributes: Dict[str, Tuple[str, str]]):
    """Wrap a module with deprecated attributes.

    Args:
        module: The module to wrap.
        deprecated_attributes: A dictionary from attribute name to pair of
            strings, where the first string gives the version that the attribute
            will be removed in, and the second string describes what the user
            should do instead of accessing this deprecated attribute.

    Returns:
        Wrapped module with deprecated attributes.
    """

    class Wrapped:
        def __getattr__(self, name):
            if name in deprecated_attributes:
                version, fix = deprecated_attributes[name]
                warnings.warn(
                    f'{name} was used but is deprecated.\n'
                    f'It will be removed in '
                    f'openfermioncirq {version}.\n'
                    f'{fix}\n',
                    DeprecationWarning,
                    stacklevel=2)
            return getattr(module, name)

    return Wrapped()
