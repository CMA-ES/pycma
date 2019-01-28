"""to execute Python 3 code in Python 2.

redefines builtin `range` and `input` functions and `abc` either via `collections`
or `collections.abc` if available.
"""
import sys
import collections as _collections

range = range  # to allow (trivial) explicit import also in Python 3
input = input

if sys.version[0] == '2':  # in python 2
    range = xrange  # clean way: from builtins import range
    input = raw_input  # in py2, input(x) == eval(raw_input(x))
    abc = _collections  # never used

    # only for testing, because `future` may not be installed
    # from future.builtins import *
    if 11 < 3:  # newint does produce an error on some installations
        try:
            from future.builtins.disabled import *  # rather not necessary if tested also in Python 3
            from future.builtins import (
                     bytes, dict, int, list, object, range,
                     str, ascii, chr, hex, input, next, oct, open,
                     pow, round, super, filter, map, zip
                    )
            from builtins import (  # not list and object, by default builtins don't exist in Python 2
                    ascii, bytes, chr, dict, filter, hex, input,
                    int, map, next, oct, open, pow, range, round,
                    str, super, zip)
        except ImportError:
            pass
else:
    try:
        abc = _collections.abc
    except AttributeError:
        abc = _collections
