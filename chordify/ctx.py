#  Copyright 2020 Matúš Škerlík
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this
#  software and associated documentation files (the "Software"), to deal in the Software
#  without restriction, including without limitation the rights to use, copy, modify, merge,
#  publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#  to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
#  OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#  OTHER DEALINGS IN THE SOFTWARE.
#

#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this
#  software and associated documentation files (the "Software"), to deal in the Software
#  without restriction, including without limitation the rights to use, copy, modify, merge,
#  publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#  to whom the Software is furnished to do so, subject to the following conditions:
#
#
#
from collections import ChainMap
from enum import Enum
from functools import partial

from .exceptions import IllegalStateError

_ctx_err_msg = """\
Working outside of application context.

This typically means that you attempted to use functionality that needed
to interface with the current application object in some way. To solve
this, set up an application context with app.app_context().  See the
documentation for more information.\
"""


def _lookup_app_object(name):
    top = _ctx_stack.top
    if top is None:
        raise RuntimeError(_ctx_err_msg)
    return getattr(top, name)


class Storage(object):
    __slots__ = "__storage__"

    def __init__(self) -> None:
        object.__setattr__(self, "__storage__", {})

    def __getattr__(self, name):
        try:
            return self.__storage__[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self.__storage__[name] = value

    def __delattr__(self, name):
        try:
            del self.__storage__[name]
        except KeyError:
            raise AttributeError(name)

    def __iter__(self):
        return iter(self.__storage__.items())


class Stack(object):
    _storage: Storage

    def __init__(self) -> None:
        super().__init__()
        self._storage = Storage()

    def push(self, obj):
        """Pushes a new item to the stack"""
        rv = getattr(self._storage, "stack", None)
        if rv is None:
            self._storage.stack = rv = []
        rv.append(obj)
        return rv

    def pop(self):
        """Removes the topmost item from the stack, will return the
        old value or `None` if the stack was already empty.
        """
        stack = getattr(self._storage, "stack", None)
        if stack is None:
            return None
        else:
            return stack.pop()

    @property
    def top(self):
        """The topmost item on the stack.  If the stack is empty,
        `None` is returned.
        """
        try:
            return self._storage.stack[-1]
        except (AttributeError, IndexError):
            return None


class ContextAttribute(object):
    """Makes an attribute forward to the config"""

    def __init__(self, name, get_converter=None):
        self.__name__ = name
        self.get_converter = get_converter

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        rv = _ctx_stack.top[self.__name__]
        if self.get_converter is not None:
            rv = self.get_converter(rv)
        return rv


class State(Enum):
    UNINITIALIZED = 0
    LEARNING = 1
    PREDICTING = 2


class Context(object):
    def __init__(self, app, client_config=None) -> None:
        super().__init__()

        self.app = app
        self.state = State.UNINITIALIZED
        self.config = ChainMap(client_config or {}, app.default_config)

        self.audio_processing = None
        self.chord_recognition = None
        self.plotter = None

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __enter__(self):
        self.push(State.PREDICTING)
        return self.app

    def __exit__(self, exc_type, exc_value, tb):
        self.pop()

    def push(self, state: State):
        del self.chord_recognition

        if state == State.UNINITIALIZED:
            raise IllegalStateError

        self.state = state
        _ctx_stack.push(self)

        self.audio_processing = self.config["AUDIO_PROCESSING_CLASS"].factory(self)
        if state == State.PREDICTING:
            self.chord_recognition = self.config["CHORD_RECOGNITION_CLASS"].factory(self)
        else:
            self.chord_recognition = self.config["CHORD_LEARNING_CLASS"].factory(self)
        self.plotter = self.config["PLOT_CLASS"].factory(self)

    def pop(self):
        _ctx_stack.pop()


_ctx_stack = Stack()
_ctx_state = partial(_lookup_app_object, "state")
_chord_processing = partial(_lookup_app_object, "chord_processing")
_chord_recognition = partial(_lookup_app_object, "chord_recognition")
_chord_resolution = partial(lambda t: object.__getattribute__(_chord_recognition(), t), "resolution")
