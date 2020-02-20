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
from contextlib import contextmanager
from functools import partial

from chordify.config import ImmutableDict
from chordify.logger import log
from chordify.state import AppState
from .exceptions import IllegalStateError, IllegalArgumentError

_ctx_err_msg = """\
Working outside of application context.

This typically means that you attempted to use functionality that needed
to interface with the current application object in some way. To solve
this, set up an application context with app.app_context().  See the
documentation for more information.\
"""


def ctx_state(mode):
    def wrap(func):
        def wrapped(*args, **kwargs):
            if _ctx_state() & mode:
                return func(*args, **kwargs)
            else:
                raise IllegalStateError("No context state information !")

        return wrapped

    return wrap


def _lookup_app_stateless(name):
    top = _ctx_stack.top
    if top is None:
        raise RuntimeError(_ctx_err_msg)
    return getattr(top, name)


@ctx_state(AppState.PREDICTING | AppState.LEARNING)
def _lookup_app(name):
    return _lookup_app_stateless(name)


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
        elif len(stack) == 1:
            return stack[-1]
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


class ConfigAttribute(object):
    """Makes an attribute forward to the config"""

    def __init__(self, name, get_converter=None):
        self.__name__ = name
        self.get_converter = get_converter

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        rv = _ctx_stack.top.config[self.__name__]
        if self.get_converter is not None:
            rv = self.get_converter(rv)
        return rv


class Context(object):
    def __init__(self, app, client_config=None) -> None:
        log(self.__class__, "Init")
        super().__init__()

        self.app = app
        self.state = AppState.UNINITIALIZED

        if _ctx_stack.top is not None:
            self.config = ImmutableDict(ChainMap(client_config or {}, _ctx_stack.top.config or {}, app.default_config))
        else:
            self.config = ImmutableDict(ChainMap(client_config or {}, app.default_config))

        self.audio_processing = None
        self.chord_recognition = None
        self.plotter = None
        self.handled_by_manager = False

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __enter__(self):
        self.push()
        self.handled_by_manager = True
        return self.app

    def __exit__(self, exc_type, exc_value, tb):
        self.handled_by_manager = False
        self.pop()

    @contextmanager
    def provide_file(self, full_path: str, mode='rb'):
        log(self.__class__, "Opening file = " + str(full_path))
        file = open(full_path, mode)
        try:
            yield file
        finally:
            log(self.__class__, "Closing file = " + str(full_path))
            file.close()

    def transition_to(self, state: AppState):
        if state == AppState.UNINITIALIZED:
            raise IllegalArgumentError

        if self.state != state:
            self.state = state
            if state == AppState.PREDICTING:
                self.audio_processing = self.config["AUDIO_PROCESSING_CLASS"].factory(self.config, self.state)
                self.chord_recognition = self.config["CHORD_RECOGNITION_CLASS"].factory(
                    self.config,
                    self.state,
                    self.provide_file("model.pickle", "rb")
                )
                self.plotter = self.config["PLOT_CLASS"].factory(self.config, self.state)
            elif state == AppState.LEARNING:
                self.audio_processing = self.config["AUDIO_PROCESSING_CLASS"].factory(self.config, self.state)
                self.chord_recognition = self.config["CHORD_LEARNING_CLASS"].factory(
                    self.config,
                    self.state,
                    self.provide_file("model.pickle", "wb")
                )
                self.plotter = self.config["PLOT_CLASS"].factory(self.config, self.state)
            else:
                raise IllegalArgumentError

    def push(self):
        if not self.handled_by_manager and self.state == AppState.UNINITIALIZED:
            if _ctx_stack.top != self:
                _ctx_stack.push(self)
                log(self.__class__, "Context pushed")
        else:
            pass

    def pop(self):
        if not self.handled_by_manager and _ctx_stack.top == self:
            log(self.__class__, "Context popped")
            _ctx_stack.pop()
        else:
            pass


_ctx_stack = Stack()
_ctx_state = partial(_lookup_app_stateless, "state")
_chord_processing = partial(_lookup_app, "chord_processing")
_chord_recognition = partial(_lookup_app, "chord_recognition")
_chord_resolution = partial(lambda t: object.__getattribute__(_chord_recognition(), t), "resolution")
