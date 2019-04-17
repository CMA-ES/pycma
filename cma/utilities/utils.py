# -*- coding: utf-8 -*-
"""various utilities not related to optimization"""
from __future__ import (absolute_import, division, print_function,
                        )  #unicode_literals, with_statement)
import os, sys, time
import warnings
from collections import defaultdict  # since Python 2.5
from collections import MutableMapping  # since Python 2.4?
import ast  # ast.literal_eval is safe eval
import numpy as np
from .python3for2 import range
del absolute_import, division, print_function  #, unicode_literals, with_statement

PY2 = sys.version_info[0] == 2
global_verbosity = 1

# array([]) does not work but np.size(.) == 0
# here is the problem:
# bool(array([0])) is False
# bool(list(array([0]))) is True
# bool(list(array([0, 1]))) is True
# bool(array([0, 1])) raises ValueError
#
# "x in emptysets" cannot be well replaced by "not x"
# which is also True for array([]) and None, but also for 0 and False,
# and False for NaN, and an exception for array([0,1]), see also
# http://google-styleguide.googlecode.com/svn/trunk/pyguide.html#True/False_evaluations
def is_(var):
    """intuitive handling of variable truth value also for `numpy` arrays.

    Return `True` for any non-empty container, otherwise the truth value of the
    scalar `var`.

    Caveat of the most unintuitive case: [0] evaluates to True, like [0, 0].

    >>> import numpy as np
    >>> from cma.utilities.utils import is_
    >>> is_({}) or is_(()) or is_(0) or is_(None) or is_(np.array(0))
    False
    >>> is_({0:0}) and is_((0,)) and is_(np.array([0]))
    True

    """
    try:  # cases: ('', (), [], {}, np.array([]))
        return True if len(var) else False
    except TypeError:  # cases None, False, 0
        return True if var else False
def is_not(var):
    """see `is_`"""
    return not is_(var)
def is_any(var_list):
    """return ``any(is_(v) for v in var_list)``"""
    return any(is_(var) for var in var_list)
def is_all(var_list):
    """return ``all(is_(v) for v in var_list)``"""
    return all(is_(var) for var in var_list)
def is_str(var):
    """`bytes` (in Python 3) also fit the bill.

    >>> from cma.utilities.utils import is_str
    >>> assert is_str(b'a') * is_str('a') * is_str(u'a') * is_str(r'b')
    >>> assert not is_str([1]) and not is_str(1)

    """
    types_ = (bytes, str)
    if PY2:
        types_ = types_ + (basestring, unicode)  # == types.StrTypes
    return any(isinstance(var, type_) for type_ in types_)
def is_nan(var):
    """return ``np.isnan(var)`` or `False` if `var` is not numeric"""
    try:
        return np.isnan(var)
    except TypeError:
        return False

def is_vector_list(x):
    """make an educated guess whether ``x`` is a list of vectors.

    >>> from cma.utilities.utils import is_vector_list as ivl
    >>> assert ivl([[0], [0]]) and not ivl([1,2,3])

    """
    try:
        return np.isscalar(x[0][0])
    except:
        return False

def as_vector_list(X):
    """a tool to handle a vector or a list of vectors in the same way,
    return a list of vectors and a function to revert the "list making".

    Useful when we might either have a single solution vector or a
    set/list/population of vectors to deal with.

    Namely, this function allows to replace a slightly more verbose::

        was_list = utils.is_vector_list(X)
        X = X if was_list else [X]
        # work work work on X, e.g.
        res = [x[0] + 1 for x in X]
        res = res if was_list else res[0]

    with::

        X, revert = utils.as_vector_list(X)
        # work work work on X, e.g.
        res = [x[0] + 2 for x in X]
        res, ... = revert(res, ...)  # also allows to revert X, if desired

    Testing:

    >>> from cma.utilities import utils
    >>> X = [3]  # a single vector
    >>> X, revert_vlist = utils.as_vector_list(X)  # BEGIN
    >>> assert X == [[3]]  # a list with one element
    >>> # work work work on X as a list of vectors, e.g.
    >>> res = [x[0] + 1 for x in X]
    >>> X, res = revert_vlist(X, res)  # END
    >>> assert res == 4
    >>> assert X[0] == 3

    """
    if is_vector_list(X):
        return X, lambda x: x
    else:
        return [X], lambda *args: args[0][0] if len(args) == 1 else (
                                    arg[0] for arg in args)

def rglen(ar):
    """return generator ``range(len(.))`` with shortcut ``rglen(.)``
    """
    return range(len(ar))

def recycled(vec, dim=None, as_=None):
    """return ``vec`` with the last element recycled to ``dim`` if
    ``len(vec)`` doesn't fail, else ``vec``.

    If ``dim`` is not given, ``len(as_)`` is used if available, else a
    scalar is returned.
    """
    try:
        len_ = len(vec)
    except TypeError:
        return vec
    if dim is None:
        try:
            dim = len(as_)
        except TypeError:
            return vec[0]
    if dim == len_:
        return vec
    elif dim < len_:
        return vec[:dim]
    elif dim > len_:
        return np.asarray(list(vec) + (dim - len_) * [vec[-1]])

def argsort(a, reverse=False):
    """return index list to get `a` in order, ie
    ``a[argsort(a)[i]] == sorted(a)[i]``
    """
    return sorted(range(len(a)), key=a.__getitem__, reverse=reverse)  # a.__getitem__(i) is a[i]

def ranks(a, reverse=False):
    """return ranks of entries starting with zero"""
    idx = argsort(a)
    return [len(idx) - 1 - idx.index(i) if reverse else idx.index(i)
            for i in range(len(idx))]

def zero_values_indices(diffs):
    """generate increasing index pairs ``(i, j)`` with ``all(diffs[i:j] == 0)``

    and ``diffs[j] != 0 or j == len(diffs)``, thereby identifying "flat
    spots/areas" in `diffs`.

    Returns the respective generator type.

    Not anymore used to smoothen ECDFs.

    Example:

    >>> from cma.utilities.utils import zero_values_indices
    >>> for i, j in zero_values_indices([0, 0.1, 0, 0, 3.2, 0, 2.1]):
    ...     print((i, j))
    (0, 1)
    (2, 4)
    (5, 6)

    """
    i = 0
    while i < len(diffs):
        if diffs[i] == 0:
            j = i
            while j < len(diffs) and diffs[j] == 0:
                j += 1
            yield i, j
            i = j + 1  # next possibly zero value
        else:
            i += 1

def pprint(to_be_printed):
    """nicely formated print"""
    try:
        import pprint as pp
        # generate an instance PrettyPrinter
        # pp.PrettyPrinter().pprint(to_be_printed)
        pp.pprint(to_be_printed)
    except ImportError:
        if isinstance(to_be_printed, dict):
            print('{')
            for k, v in to_be_printed.items():
                print("'" + k + "'" if str(k) == k else k,
                      ': ',
                      "'" + v + "'" if str(v) == v else v,
                      sep="")
            print('}')
        else:
            print('could not import pprint module, appling regular print')
            print(to_be_printed)

def num2str(val, significant_digits=2, force_rounding=False,
            max_predecimal_digits=5, max_postdecimal_leading_zeros=1,
            remove_trailing_zeros=True, desired_length=None):
    """returns the shortest string representation.

    Generally, display either ``significant_digits`` digits or its true
    value, whichever is shorter.

    ``force_rounding`` shows no more than the desired number of significant
    digits, which means, e.g., ``12345``  becomes ``12000``.

    ``remove_trailing_zeros`` removes zeros, if and only if the value is
    exactly.

    ``desired_length`` adds digits up to the desired length.

    >>> from cma.utilities import utils
    >>> print([utils.num2str(val) for val in [12345, 1234.5, 123.45,
    ...       12.345, 1.2345, .12345, .012345, .0012345]])
    ['12345', '1234', '123', '12', '1.2', '0.12', '0.012', '1.2e-3']

    """
    if val == 0:
        return '0'
    if not significant_digits > 0:
        raise ValueError('need significant_digits=%s > 0'
                         % str(significant_digits))
    is_negative = val < 0
    original_value = val
    val = float(np.abs(val))

    order_of_magnitude = int(np.floor(np.log10(val)))
    # number of digits before decimal point == order_of_magnitude + 1
    fac = 10**(significant_digits - 1 - order_of_magnitude)
    val_rounded = np.round(fac * val) / fac

    # the strategy is now to produce two string representations
    # cut each down to the necessary length and return the better

    # the first is %f format
    if order_of_magnitude + 1 >= significant_digits:
        s = str(int(val_rounded if force_rounding else np.round(val)))
    else:
        s = str(val_rounded)
        idx1 = 0  # first non-zero index
        while idx1 < len(s) and s[idx1] in ('-', '0', '.'):
            idx1 += 1  # find index of first significant number
        idx2 = idx1 + significant_digits + (s.find('.') > idx1)
        # print(val, val_rounded, s, len(s), idx1, idx2)
        # pad some zeros in the end, in case
        if val != val_rounded:
            if len(s) < idx2:
                s += '0' * (idx2 - len(s))
        # remove zeros from the end, in case
        if val == val_rounded and remove_trailing_zeros:
            while s[-1] == '0':
                s = s[0:-1]
        if s[-1] == '.':
            s = s[0:-1]
    s_float = ('-' if is_negative else '') + s

    # now the second, %e format
    s = ('%.' + str(significant_digits - 1) + 'e') % val
    if eval(s) == val and s.find('.') > 0:
        while s.find('0e') > 0:
            s = s.replace('0e', 'e')
    s = s.replace('.e', 'e')
    s = s.replace('e+', 'e')
    while s.find('e0') > 0:
        s = s.replace('e0', 'e')
    while s.find('e-0') > 0:
        s = s.replace('e-0', 'e-')
    if s[-1] == 'e':
        s = s[:-1]
    s_exp = ('-' if is_negative else '') + s

    # print(s_float, s_exp)

    # now return the better (most of the time the shorter) representation
    if (len(s_exp) < len(s_float) or
        s_float.find('0.' + '0' * (max_postdecimal_leading_zeros + 1)) > -1 or
        np.abs(val_rounded) >= 10**(max_predecimal_digits + 1)
        ):
        s_ret = s_exp
    else:
        s_ret = s_float
    if desired_length:
        s_old = ''
        while len(s_ret) < desired_length and len(s_old) < len(s_ret):
            s_old = s_ret
            s_ret = num2str(original_value,
                       significant_digits + desired_length - len(s_ret),
                       force_rounding,
                       max_predecimal_digits,
                       max_postdecimal_leading_zeros,
                       remove_trailing_zeros,
                       desired_length=None)
    return s_ret

# todo: this should rather be a class instance
def print_warning(msg, method_name=None, class_name=None, iteration=None,
                   verbose=None, maxwarns=None):
    """Poor man's maxwarns: warn only if ``iteration<=maxwarns``"""
    if verbose is None:
        verbose = global_verbosity
    if maxwarns is not None and iteration is None:
        raise ValueError('iteration must be given to activate maxwarns')
    if verbose >= -2 and (iteration is None or maxwarns is None or
                            iteration <= maxwarns):
        warnings.warn(msg + ' (' +
              ('class=%s ' % str(class_name) if class_name else '') +
              ('method=%s ' % str(method_name) if method_name else '') +
              ('iteration=%s' % str(iteration) if iteration else '') +
              ')')
def print_message(msg, method_name=None, class_name=None, iteration=None,
                   verbose=None):
    if verbose is None:
        verbose = global_verbosity
    if verbose >= 0:
        print('NOTE (module=cma' + # __name__ +
              (', class=' + str(class_name) if class_name else '') +
              (', method=' + str(method_name) if method_name else '') +
              (', iteration=' + str(iteration) if iteration is not None else '') +
              '): ', msg)

def set_attributes_from_dict(self, dict_, initial_params_dict_name=None):
    """assign, for example, all arguments given to an ``__init__``
    method to attributes in ``self`` or ``self.params`` or ``self.args``.

    If ``initial_params_dict_name`` is given, ``dict_`` is also copied
    into an attribute of ``self`` with name ``initial_params_dict_name``::

        setattr(self, initial_params_dict_name, dict_.copy())

    and the ``self`` key is removed from the copied `dict` if present.

    >>> from cma.utilities.utils import set_attributes_from_dict
    >>> class C(object):
    ...     def __init__(self, arg1, arg2, arg3=None):
    ...         assert len(locals()) == 4  # arguments are locally visible
    ...         set_attributes_from_dict(self, locals())
    >>> c = C(1, 22)
    >>> assert c.arg1 == 1 and c.arg2 == 22 and c.arg3 is None
    >>> assert len(c.__dict__) == 3 and not hasattr(c, 'self')

    Details:

    - The entry ``dict_['self']`` is always ignored.

    - Alternatively::

        self.args = locals().copy()
        self.args.pop('self', None)  # not strictly necessary

      puts all arguments into ``self.args: dict``.

    """
    if initial_params_dict_name:
        setattr(self, initial_params_dict_name, dict_.copy())
        getattr(self, initial_params_dict_name).pop('self', None)
    for key, val in dict_.items():
        if key != 'self':  # avoid self referencing
            setattr(self, key, val)

def download_file(url, target_dir='.', target_name=None):
    import urllib2
    if target_name is None:
        target_name = url.split(os.path.sep)[-1]
    with open(os.path.join(target_dir, target_name), 'wb') as f:
        f.write(urllib2.urlopen(url).read())

def extract_targz(tarname, filename=None, target_dir='.'):
    """filename must be a valid path in the tar"""
    import tarfile
    tmp_dir = '._tmp_'
    if filename is None:
        tarfile.TarFile.gzopen(tarname).extractall(target_dir)
    else:
        import shutil
        tarfile.TarFile.gzopen(tarname).extractall(tmp_dir)
        shutil.copy2(os.path.join(tmp_dir, filename),
                     os.path.join(target_dir, filename.split(os.path.sep)[-1]))
        shutil.rmtree(tmp_dir)

class BlancClass(object):
    """blanc container class to have a collection of attributes.

    For rapid shell- or prototyping. In the process of improving the code
    this class might/can/will at some point be replaced with a more
    tailored class.

    Usage:

    >>> from cma.utilities.utils import BlancClass
    >>> p = BlancClass()
    >>> p.value1 = 0
    >>> p.value2 = 1

    """

class DictClass(dict):
    """A class wrapped over `dict` to use class .-notation.

    >>> from cma.utilities.utils import DictClass
    >>> dict_ = dict((3 * c, c) for c in 'abcd')
    >>> as_class = DictClass(dict_)
    >>> assert as_class.__dict__ == dict_ == as_class
    >>> assert as_class.aaa == 'a'
    >>> as_class.new = 33
    >>> assert 'new' in as_class
    >>> as_class['nnew'] = 44
    >>> assert as_class.nnew == 44
    >>> assert len(as_class) == 6

    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
    def __dir__(self):
        return self.keys()

class DerivedDictBase(MutableMapping):
    """for conveniently adding methods/functionality to a dictionary.

    The actual dictionary is in ``self.data``. Derive from this
    class and copy-paste and modify setitem, getitem, and delitem,
    if necessary.

    Details: This is the clean way to subclass the build-in dict, however
    it depends on `MutableMapping`.

    """
    def __init__(self, *args, **kwargs):
        # MutableMapping.__init__(self)
        super(DerivedDictBase, self).__init__()
        # super(SolutionDict, self).__init__()  # the same
        self.data = dict()
        self.data.update(dict(*args, **kwargs))
    def __len__(self):
        return len(self.data)
    def __contains__(self, key):
        return key in self.data
    def __iter__(self):
        return iter(self.data)
    def __setitem__(self, key, value):
        """define ``self[key] = value``"""
        self.data[key] = value
    def __getitem__(self, key):
        """define access ``self[key]``"""
        return self.data[key]
    def __delitem__(self, key):
        del self.data[key]

class SolutionDict(DerivedDictBase):
    """dictionary with computation of an hash key.

    The hash key is generated from the inserted solution and a stack of
    previously inserted same solutions is provided. Each entry is meant
    to store additional information related to the solution.

        >>> import cma.utilities.utils as utils, numpy as np
        >>> d = utils.SolutionDict()
        >>> x = np.array([1,2,4])
        >>> d[x] = {'f': sum(x**2), 'iteration': 1}
        >>> assert d[x]['iteration'] == 1
        >>> assert d.get(x) == (d[x] if d.key(x) in d.keys() else None)

    TODO: data_with_same_key behaves like a stack (see setitem and
    delitem), but rather should behave like a queue?! A queue is less
    consistent with the operation self[key] = ..., if
    self.data_with_same_key[key] is not empty.

    TODO: iteration key is used to clean up without error management

    """
    def __init__(self, *args, **kwargs):
        # DerivedDictBase.__init__(self, *args, **kwargs)
        super(SolutionDict, self).__init__(*args, **kwargs)
        self.data_with_same_key = {}
        self.last_iteration = 0
    def key(self, x):
        """compute key of ``x``"""
        try:
            return tuple(x)
            # using sum(x) is slower, using x[0] is slightly faster
        except TypeError:
            return x
    def __setitem__(self, key, value):
        """define ``self[key] = value``"""
        key = self.key(key)
        if key in self.data_with_same_key:
            self.data_with_same_key[key] += [self.data[key]]
        elif key in self.data:
            self.data_with_same_key[key] = [self.data[key]]
        self.data[key] = value
    def __getitem__(self, key):  # 50% of time of
        """define access ``self[key]``"""
        return self.data[self.key(key)]
    def __delitem__(self, key):
        """remove only most current key-entry of list with same keys"""
        key = self.key(key)
        if key in self.data_with_same_key:
            if len(self.data_with_same_key[key]) == 1:
                self.data[key] = self.data_with_same_key.pop(key)[0]
            else:
                self.data[key] = self.data_with_same_key[key].pop(-1)
        else:
            del self.data[key]
    def truncate(self, max_len, min_iter):
        """delete old entries to prevent bloat"""
        if len(self) > max_len:
            for k in list(self.keys()):
                if self[k]['iteration'] < min_iter:
                    del self[k]
                    # deletes one item with k as key, better delete all?

class DataDict(defaultdict):
    """a dictionary of lists (of data)"""
    def __init__(self, filename='_data.py'):
        self.filename = filename
        defaultdict.__init__(self, list)
        self.load()

    def load(self):
        """element-wise append/merge data of loaded `dict` to self,

        by calling `update`.

        To load cleanly without merge use `clear` + `load` or the class
        constructor with a new `filename`.
        """
        with open(self.filename, 'rt') as f:
            dd = ast.literal_eval(f.read())
        self.update(dd)
        return self

    def update(self, dict_):
        """append data of entries in `dict_` to entries in self"""
        for k in dict_:
            self[k] += dd[k]  # self is a dict of lists
        return self

    def save(self):
        with open(self.filename, 'wt') as f:
            f.write(repr(dict(self)))

    def clear(self):
        for key in [k for k in self]:
            del self[key]
        return self

class ExclusionListOfVectors(list):
    """For delayed selective mirrored sampling"""
    def __contains__(self, vec):
        for v in self:
            if 1 - 1e-9 < np.dot(v, vec) / (sum(np.asarray(v)**2) * sum(np.asarray(vec)**2))**0.5 < 1 + 1e-9:
                return True
        return False

class ElapsedWCTime(object):
    """measure elapsed cumulative time while not paused and elapsed time
    since last tic.

    Use attribute `tic` and methods `pause` () and `reset` ()
    to control the timer. Use attributes `toc` and `elapsed` to see
    timing results.

    >>> import cma
    >>> e = cma.utilities.utils.ElapsedWCTime().pause()  # (re)start later
    >>> assert e.paused and e.elapsed == e.toc < 0.1
    >>> assert e.toc == e.tic < 0.1  # timer starts here
    >>> assert e.toc <= e.tic  # toc is usually a few microseconds smaller
    >>> assert not e.paused    # the timer is now running due to tic

    Details: the attribute ``paused`` equals to the time [s] when paused or
    to zero when the timer is running.
    """
    def __init__(self, time_offset=0):
        """add time offset in seconds and start timing"""
        self._time_offset = time_offset
        self.reset()
    def reset(self):
        """reset to initial state and start timing"""
        self.cum_time = self._time_offset
        self.paused = 0
        """time when paused or 0 while running"""
        self.last_tic = time.time()
        return self
    def pause(self):
        """pause timer, resume with `tic`"""
        if not self.paused:
            self.paused = time.time()
        return self
    def __call__(self):
        """depreciated return elapsed time (for backwards compatibility)
        """
        raise DeprecationWarning()
        return self.elapsed
    @property
    def tic(self):
        """return `toc` and restart tic/toc last-round-timer.

        In case, also resume from `pause`.
        """
        return_ = self.toc
        if self.paused:
            if self.paused < self.last_tic:
                print_warning("""paused time=%f < last_tic=%f, which
                should never happen, but has been observed at least once.
                """ % (self.paused, self.last_tic),
                              "tic", "ElapsedWCTime")
                self.paused = self.last_tic
            self.cum_time += self.paused - self.last_tic
        else:
            self.cum_time += time.time() - self.last_tic
        self.paused = 0
        self.last_tic = time.time()
        return return_
    @property
    def elapsed(self):
        """elapsed time while not paused, measured since creation or last
        `reset`
        """
        return self.cum_time + self.toc
    @property
    def toc(self):
        """return elapsed time since last `tic`"""
        if self.paused:
            return self.paused - self.last_tic
        return time.time() - self.last_tic

class TimingWrapper(object):
    """wrap a timer around a callable.

    Attribute ``timer`` collects the timing data in an `ElapsedWCTime`
    class instance, in particular the overall elapsed time in
    ``timer.elapsed`` and the time of the last call in ``timer.toc``.
    """
    def __init__(self, callable_):
        """``callable_`` is the `callable` to be timed when called"""
        self._callable = callable_
        self.timer = ElapsedWCTime().pause()
    def __call__(self, *args, **kwargs):
        self.timer.tic
        res = self._callable(*args, **kwargs)
        self.timer.pause()
        return res

class DictFromTagsInString(dict):
    """read from a string or file all key-value pairs within all
    ``<python>...</python>`` tags and return a `dict`.

    Within the tags valid Python code is expected: either a list of
    key-value pairs ``[[key1, value1], [key2, value2], ...]`` or a
    dictionary ``{ key1: value1, key2: value2, ...}``. A key can be any
    immutable object, while it is often a string or a number.

    The `as_python_tag` attribute provides the respective (tagged) string.
    The ``tag_string`` attribute defines the tag identifier, 'python' by
    default, and can be change if desired at any time.

    >>> from cma.utilities.utils import DictFromTagsInString
    >>> s = '<python> [[33, 44], ["annotations", [None, 2]]] </python>'
    >>> s += '<python> {"annotations": [2, 3]} </python>'
    >>> d = DictFromTagsInString(s)
    >>> # now d.update can be used to read more tagged strings/files/...
    >>> assert d.tag_string == 'python'  # can be set to any other value
    >>> d.tag_string = 'pyt'
    >>> # now 'pyt' tags can/will be read (only)
    >>> assert str(d).startswith('<pyt>{') and str(d).endswith('}</pyt>')
    >>> assert len(d) == 2 and d[33] == 44 and d['annotations'] == [2, 3]

    When the same key appears several times, its value is overwritten.
    """
    def __init__(self, *args, **kwargs):
        """for input args see `update` method."""
        super(DictFromTagsInString, self).__init__()  # not necessary!?
        self.tag_string = "python"
        if is_(args) or is_(kwargs):
            self.update(*args, **kwargs)
    def update(self, string_=None, filename=None, file_=None, dict_=None,
               tag_string=None):
        """only one of the first four arguments is accepted at a time,
        return ``self``.

        If the first argument has no keyword, it is assumed to be a string
        to be parsed for tags.
        """

        args = 4 - ((string_ is None) + (filename is None) +
               (file_ is None) + (dict_ is None))
        if not args:
            raise ValueError('''nothing to update''')
        if args > 1:
            raise ValueError('''
                use either string_ or filename or file_ or dict_ as
                input, but not several of them''')
        if tag_string is not None:
            self.tag_string = tag_string
        if filename is not None:
            string_ = open(filename, 'r').read()
        elif file_ is not None:
            string_ = file_.read()
        elif dict_ is not None:
            super(DictFromTagsInString,
                  self).update(dict_)
            return self
        super(DictFromTagsInString,
              self).update(self._eval_python_tag(string_))
        return self
    @property
    def as_python_tag(self):
        return self._start + repr(dict(self)) + self._end
    def __repr__(self):
        return self.as_python_tag
    @property
    def _start(self):
        return '<' + self.tag_string + '>'
    @property
    def _end(self):
        return '</' + self.tag_string + '>'
    def _eval_python_tag(self, str_):
        """read [key, value] pairs from a `list` or a `dict` within all
        ``<self.tag_str>`` tags in ``str_`` and return a `dict`.

        >>> from cma.utilities.utils import DictFromTagsInString
        >>> s = '<py> [[33, 44], ["annotations", []]] </py>'
        >>> s += '<py>[["annotations", [1,2,3]]] </py>'
        >>> d = DictFromTagsInString()
        >>> assert len(d) == 0
        >>> d.update(s)  # still empty as default tag is not <py>
        <python>{}</python>
        >>> assert len(d) == 0
        >>> d.tag_string = "py"  # set desired tag
        >>> d.update(s)  # doctest:+ELLIPSIS
        <py>{...
        >>> assert len(d) == 2
        >>> assert d[33] == 44 and len(d["annotations"]) == 3

        """
        values = {}
        str_lower = str_.lower()
        start = str_lower.find(self._start)
        while start >= 0:
            start += len(self._start)  # move behind begin tag
            end = str_lower.find(self._end, start)
            values.update(ast.literal_eval(str_[start:end].strip()))
            start = str_lower.find(self._start, start + 1)
        return values

class MoreToWrite(list):
    """make sure that this list does not grow unbounded"""
    def __init__(self):
        self._lenhist = []
    def check(self):
        self._lenhist += [len(self)]
        if len(self._lenhist) > 3:
            if all(np.diff(self._lenhist) > 0):
                del self[:]
            self._lenhist = []

class DefaultSettings(object):
    """resembling somewhat `types.SimpleNamespace` from Python >=3.3
    but with instantiation and resembling even more the `dataclass` decorator
    from Python >=3.7.

    ``MyClassSettings(DefaultSettings)`` is preferably used by assigning a settings
    attribute in ``__init__`` like:

    >>> class MyClass:
    ...     def __init__(self, a, b=None, param1=None, c=3):
    ...         self.settings = MyClassSettings(locals(), 1, self)

    The `1` signals, purely for consistency checking, that one parameter defined
    in ``MyClassSettings`` is to be set from ``locals()``. ``MyClassSettings``
    doesn't use any names which are already defined in ``self.__dict__``. The
    settings are defined in a derived parameter class like

    >>> from cma.fitness_models import DefaultSettings
    >>> class MyClassSettings(DefaultSettings):
    ...     param1 = 123
    ...     val2 = False
    ...     another_par = None  # we need to assign at least None always

    The main purpose is, with the least effort, (i) to separate
    parameters/settings of a class from its remaining attributes, and (ii) to be
    flexible as to which of these parameters are arguments to ``__init__``.
    Parameters can always be modified after instantiation. Further advantages
    are (a) no typing of ``self.`` to assign the default value or the passed
    parameter value (the latter are assigned "automatically") and (b) no
    confusing name change between the passed option and attribute name is
    possible.

    The class does not allow to overwrite the default value with `None`.

    Now any of these parameters can be used or re-assigned like

    >>> c = MyClass(0.1)
    >>> c.settings.param1 == 123
    True
    >>> c = MyClass(2, param1=False)
    >>> c.settings.param1 is False
    True

    """
    def __init__(self, params, number_of_params, obj):
        """Overwrite default settings in case.

        :param params: A dictionary (usually locals()) containing the parameters to set/overwrite
        :param number_of_params: Number of parameters to set/overwrite
        :param obj: elements of obj.__dict__ are in the ignore list.
        """
        self.inparams = dict(params)
        self._number_of_params = number_of_params
        self.obj = obj
        self.inparams.pop('self', None)
        self._set_from_defaults()
        self._set_from_input()

    def __str__(self):
        # return str(self.__dict__)
        return ("{" + '\n'.join(r"%s: %s" % (str(k), str(v)) for k, v in self.items()) + "}")

    def _set_from_defaults(self):
        """defaults are taken from the class attributes"""
        self.__dict__.update(((key, val)
                              for (key, val) in type(self).__dict__.items()
                              if not key.startswith('_')))
    def _set_from_input(self):
        """Only existing parameters/attributes and non-None values are set.

        The number of parameters is cross-checked.

        Remark: we could select only the last arguments
        of obj.__init__.__func__.__code__.co_varnames
        which have defaults obj.__init__.__func__.__defaults__ (we do
        not need the defaults)
        """
        discarded = {}  # discard name if not in self.__dict__
        for key in list(self.inparams):
            if key not in self.__dict__ or key in self.obj.__dict__:
                discarded[key] = self.inparams.pop(key)
            elif self.inparams[key] is not None:
                setattr(self, key, self.inparams[key])
        if len(self.inparams) != self._number_of_params:
            warnings.warn("%s: %d parameters desired; remaining: %s; discarded: %s "
                          % (str(type(self)), self._number_of_params, str(self.inparams),
                             str(discarded)))
        # self.__dict__.update(self.inparams)
        delattr(self, 'obj')  # prevent circular reference self.obj.settings where settings is self

