"""Tools for (lean) experimentation and collecting data.

See also cma.optimization_tools and cma.utilities.math.test*.

TODO: write some running examples as doctests.
"""
from __future__ import division, print_function
import warnings
import os
import shutil
import time
from collections import defaultdict, namedtuple
from ast import literal_eval
import math
import numpy as np  # also to replace 'inf' with np.inf and vice versa
from cma import warnings_and_exceptions as _warnings_and_exceptions


def unique_time_stamp(decimals=2):
    "a unique timestamp up to 1/10^decimals seconds"
    s = '{0:.' + str(decimals) + 'f}' + '.' * (decimals == 0)
    return time.strftime("%Y-%m-%d-%Hh%Mm%Ss") + (
            s.format(time.time()).split('.')[1])

def down_sample(x, y=None, len_=500):
    """return (index, x_down) if y is None, else (x_down, y_down).

    Example: ``plot(*down_sample(mydata))``

    """
    if y is None:
        x, y = np.arange(len(x)), np.asarray(x)
    while len(y) > 2 * len_:
        x = x[::2]
        y = y[::2]
    return x, y

def sp1(data):
    """return the average of finite entries in `data`,

    multiplied by the ratio ``Ndata / Nfinite == len(data) /
    sum(np.isfinite(data))``.

    This measure has been called Q-measure or success performance one, sp1. It
    is computed as the average over all finite entries times the number of all
    entries divided by the number of finite entries. If all entries are finite
    this is the average value.

    Details: the multiplier operates under the assumption that non-finite
    entries contribute the same as the average finite entry and resampling can
    be applied to get a finite entry.
    """
    a = np.asarray(data)
    idx = np.isfinite(a)
    return len(a) * np.mean(a[idx]) / sum(idx) if sum(idx) else np.inf

def copy_file(src, dest):
    """copy src to dest and created dest folder(s) if necessary"""
    if not os.path.exists(src):
        warnings.warn("{0} does not exist (yet), nothing to be copied"
                     .format(src))
        return
    try:
        os.makedirs(os.path.dirname(dest))  # bails when a folder exists
    except Exception:  # error or IOError or OSError or FileExistsError?
        pass
    shutil.copy2(src, dest)

class TimeWarner:
    """context manager to print the time spent iff it exceeds a threshold.

    Usage: ``with TimeWarner('saving'):``, the argument is used only for
    printing the message in case.
"""
    def __init__(self, name, warn_time=2):
        self.name = name
        self.warn_time = warn_time
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.t1 = time.time()
        if self.t1 - self.t0 > self.warn_time:
            print("TimeWarner: {} took {:.1f} seconds".format(self.name, self.t1 - self.t0))

class ClassFromDict(object):
    """set class attributes from a `dict`, see also `cma.utilities.utils.DictClass2`"""
    def __init__(self, dict_):
        self._dict = dict(dict_)
        for key in dict_:
            setattr(self, key, dict_[key])
    @property
    def as_dict(self):
        """collect only original attributes, check out ``.__dict__`` to see also
        the attributes later added.
        """
        return dict([key, getattr(self, key)] for key in self._dict)

class Results(object):
    """a container to communicate (changing) results via disk.

    The ``data`` attribute is a dictionary (`DataDict`) which contains all
    observed data, usually a list for each key. Any new container reads
    previously saved data (with the same name) on initialization.

    Before the data are saved, the old data are backupped under the current
    timestamp.

    Use case example::

        import cma.experimentation

        # create or load 'sweep_data' results
        res = cma.experimentation.Results('sweep_data')

        # save data after each trial
        for dim in dimensions:
            res.backup()  # in case we want fewer backups
            for ...:
                x, es = cma.fmin2(...)
                res.data[dim] += [es.result.evaluations if 'ftarget' in es.result.stop()
                                  else np.inf]
                res.save(backup=False)  # like this we can never lose data


    Details: Saving/loading of nonfinite values with `ast.literat_eval` is
    covered with the ``values_to_string`` parameters.

    Some methods handle nested dictionaries, however it seems to be safer to use
    tuples as keys in a flat dictionary or use different `Results` instances,
    e.g. by dimension and/or popsize, like::

        res = experimentation.Results('sweep_c_dim={}lam={}.format(dimension, popsize)

    In particular ``.data.clean()`` for `float` keys does (currently) not work
    with key tuples.
"""
    def __init__(self,
                 name=None,
                 values_to_string = (np.inf, np.nan, math.nan),
                 verbose=1):
        self.filename = 'results_dict' if name is None else name
        if '.' not in self.filename[-8:]:
            self.filename = self.filename + '.pydict'
        path, name = os.path.split(self.filename)
        name, ext = os.path.splitext(name)
        self._extension = ext
        self.backup_dirname = os.path.join(path, '._' + name + '-backup')
        self._values_to_string_vals = values_to_string
        self.data = DataDict()  # {}
        self.meta_data = dict(info="", )
        if os.path.exists(self.filename):
            self.load()
            if self.data and verbose > 0:
                keys = sorted(self.data)
                print('Results("{3}") loaded {0} data key entries ({1} ... {2})'
                      ' with {4} values'
                      .format(len(self.data), keys[0], keys[-1],
                              self.filename, sum(self.samples().values())))

    def __getattr__(self, name):
        """access ``Results.data`` directly from `Results`"""
        return getattr(self.data, name)

    def _load_json(self):
        """return dictionary from load json file with keys to `int` or `float`, if possible"""
        import json
        filename = self.filename.rsplit('.', 1)[0] + '.json'
        try:
            res = json.load(filename)
        except:
            warnings.warn('json file not loaded, this warning needs to be improved/removed')
            return
        raise NotImplementedError('how do we know whether to convert keys into numbers?')
        for k in sorted(res):  # sort is not needed or effective for now
            for to_number in (int, float):
                try:
                    knew = to_number(k)
                except ValueError:
                    pass
                else:
                    res[knew] = res[k]
                    del res[k]
                    break
        # self.data = DataDict(res)  # this is the idea, then still values need to be fixed
        return res

    def load(self):
        """load data from file, discard current data, see also `update`"""
        if self.data:
            self.backup()
        with open(self.filename, 'rt') as f:
            try:
                self.data = DataDict(literal_eval(f.read()))
            except Exception:
                print("Please check whether inf or nan were used as simple value"
                      " rather than in a sequence as [inf] or [nan].")
                raise
        self._values_to_string(inverse=True)
        self.meta_data = self.data.pop('meta_data', None)
        return self

    def update(self, filename, backup=True):
        """Append data from ``filename`` to ``self.data``.
        
        To update self with a ``data`` `dict` instead a file, call
        ``self.data.update(data)``.

        Details: assumes a flat `dict` structure.
        """
        data = Results(filename).data
        if backup:
            self.backup()
        for key in data:
            self.data[key] += data[key]
        return self

    def save(self, backup=True):
        """save `self.data` to disk"""
        if backup:
            self.backup()
            # try:
            #     with open(self.filename, 'rt') as f:
            #         self.backup(f.read())
            # except IOError: pass
        self._values_to_string()
        if self.data.get('meta_data', None) is None and self.meta_data is not None:
            self.data['meta_data'] = repr(self.meta_data)
        try:
            os.makedirs(os.path.dirname(self.filename))  # bails when a folder exists
        except Exception:  # error or IOError or OSError or FileExistsError?
            pass
        with open(self.filename, 'wt') as f:
            f.write(repr(self.data))
        self.data.pop('meta_data')  # otherwise they get too often in the way
        self._values_to_string(inverse=True)
        return self  # necessary/useful?

    def _value_to_string(self, val, inverse=False):
        """return "correct" value, doesn't work for `nan`"""
        if inverse:
            for val_ in self._values_to_string_vals:
                if repr(val_) == val:
                    return val_
            return val
        if val in self._values_to_string_vals:  # works with nan too
            return repr(val)
        return val
    def _values_to_string(self, inverse=False):
        """replace certain values in lists with strings or vice versa.

        Values to be replaced are defined at instance creation in
        `_values_to_string_vals`.

        Details: this prevents `ast.literal_eval` to bail on, for example,
        ``repr([1, 3, 'inf'])``, as 'inf' is just a string and will not be
        evaluated.
        """

        for a in self._arrays():
            try: len(a), a[0]
            except (TypeError, IndexError):
                try:
                    literal_eval(repr(a))
                except:
                    warnings.warn(
                        "non-sequence found as data value in {0}, which will fail when"
                        "\n  reading back serialized data. A simple fix is to use"
                        "\n  ``[val]`` instead of ``val`` or avoid values which"
                        "\n  `ast.literal_eval` cannot digest, like ``inf``, ``nan``, etc."
                .format(a))
            else:
                for i in range(len(a)):
                    a[i] = self._value_to_string(a[i], inverse)

        for val in self._values_to_string_vals:
            new, old = repr(val), val
            if inverse:
                new, old = old, new
            if old in self.data:
                self.data[new] = self.data[old]
                del self.data[old]

    def _arrays(self, data=None):
        """return a flat list of (references to) all non-dicts in data.

        Traverses recursively down into dictionaries.
        """
        if data is None:
            data = self.data
        if isinstance(data, dict):
            res = []
            for key in data:  # travers down
                if key != 'meta_data':
                    res += self._arrays(data[key])
            return res
        else:  # return leave
            return [data]

    def backup(self):
        """backup saved data by making a file copy"""
        dest = os.path.join(self.backup_dirname,
                            unique_time_stamp(2) + self._extension)
        copy_file(self.filename, dest)
        self.last_backup = dest

    def _backup0(self, data=None):
        """append data to backup file, no easy to read back"""
        if data is None:
            self._values_to_string()
            datastring = repr(self.data)
            self._values_to_string(inverse=True)
        else:
            datastring = repr(data)
        with open(self.backup_dirname + self._extension, 'at') as f:
            f.write(datastring)
            f.write('\n')

class DataDict(defaultdict):
    """A (default) dictionary like ``parameter_value: list_of_measures``,

    e.g. with float parameter or dimension as key and runtimes as value. This
    class is used under the hood in the `Results` class.

    A main functionality is the method `clean`, which joins all entries
    which have almost equal keys. This allows to have a `float` parameter
    as key.

    This class provides simple computations on this kind of data,
    like ``x, y = .xy_arrays() == sorted(keys), sp1(values)``.

    If the dictionary values are not lists, one may get rather unexpected
    results or exceptions.

    Details: this class allows to use `float` values as keys when
    `clean_key` and `set_clean` are used to access the data in the
    `dict`. Inheriting from `defaultdict` with `list` as default value,
    the syntax::

        data = DataDict()
        data[first_key] += [first_data_point]

    without initialization of the key value works perfectly fine.

    Caveat: small values are considered as the same key, even if they are
    close to zero. Either use a different comparison via the `equal`
    keyword parameter, or use ``1 / key_value`` or `log(key_value)``.

    TODO: consider `numpy.allclose` for almost equal comparison?
"""
    def __init__(self, dict_=None):
        """Use ``dict(dict_.data or dict_)``, and `dict_.meta_data` for
        initialization.

        Details: `dict_.meta_data` are assigned as a reference.
        """
        defaultdict.__init__(self, list)
        if dict_ is not None:
            if hasattr(dict_, 'meta_data'):
                self.meta_data = dict_.meta_data
            if hasattr(dict_, 'data'):
                self.update(dict_.data)
            else:
                self.update(dict_)

    def update(self, dict_):
        """update data lists from a `dict` of lists (and only a `dict`)"""
        data = self
        for k in dict_:
            data[k] += dict_[k]

    def xy_arrays(self, agg=sp1, type_=np.asarray):
        """return two arrays ready to be plotted like ``plot(*xy_arrays)``.

        The x-array contains the sorted keys, the y-array contains the
        respectively aggregated values.

        For example to be used like::

            ``plot(*self.xy_arrays())``.

        Parameter `agg` determines the function to aggregate data values, by
        default `sp1` which is the mean corrected for missing data. To show
        dispersion, we can use ``agg=lambda x: np.percentile(x, 10)`` and ``...,
        90)``.
        """
        data = self
        keys = sorted(data)   # [k for k in sorted(self) if np.isscalar(k)]
        return (type_(keys),
                type_([agg(data[k]) for k in keys]))

    def aggregated(self, agg=sp1, relative=False, by=None):
        """WIP return a `dict` with ``(agg(data), number_of_samples)`` per key.

        For example, to get the 10%tile::

            .aggregated(lambda x: np.percentile(x, 10))

        """
        data = self
        def normalize_in_place(res):
            min_ = min(v[0] for v in res.values())
            for k in res:
                res[k] = res[k][0] / min_, res[k][1]
            return res

        keys = sorted(data)  # was: [k for k in sorted(data) if np.isscalar(k)]
        res = {k: (float(agg(data[k])), len(data[k])) for k in keys}
        if relative:
            if by is None:
                normalize_in_place(res)
            else:
                by_values = [k[by] for k in res]
                res_ = {}
                for by_ in sorted(set(by_values)):
                    res_[by_] = normalize_in_place({k: v for k, v in res.items() if k[by] == by_})
                return res_
        return res
    
    def samples(self):
        """return a `dict` with the number of values per key"""
        data = self
        return {k: len(data[k]) for k in sorted(data)}

    def argmin(self, agg=sp1, slack=1.0, slack_index_shift=+1):
        """slack_index_shift can be +-1, looking to the right/left of argmin"""
        x, y = self.xy_arrays(agg)
        idxmin = np.argmin(y)
        if not slack_index_shift:
            return x[idxmin]
        assert slack_index_shift in (-1, 1), slack_index_shift  # quick hack
        i = idxmin + slack_index_shift
        while i >= 0 and i < len(y) and y[i] <= slack * y[idxmin]:
            i += slack_index_shift
        return x[i - slack_index_shift]

    def tests(self, gap=1, method='auto'):
        """return p-values of the `mannwhitneyu` test of entries adjacent with `gap`"""
        import scipy.stats
        data = self
        keys = sorted(data)
        return {(keys[i], keys[i+gap]): float(
                    scipy.stats.mannwhitneyu(
                        data[keys[i]], data[keys[i+gap]],
                        method=method, alternative='two-sided')[1])
                for i in range(len(keys) - gap)}
    def test(self, key, key2=None, method='auto'):
        """return p-value of the `mannwhitneyu` test"""
        import scipy.stats
        data = self
        keys = sorted(data)
        if key not in keys:
            raise ValueError("key {} not in data keys {}".format(key, keys))
        if key2 is None:
            idx = keys.index(key) + 1
            if idx >= len(keys):
                raise ValueError("key {} is the last key in data {}".format(key, keys))
            key2 = keys[idx]
        return {(key, key2): float(
                    scipy.stats.mannwhitneyu(
                        data[key], data[key2],
                        method=method, alternative='two-sided')[1])}

    def clean(self, equal=lambda x, y: x - 1e-6 < y < x + 1e-6):
        """merge keys which have almost the same value"""
        for key in list(self.keys()):
            if key not in self:  # self.keys() changes in the process
                continue
            self.clean_key(key, equal=equal)
        return self

    def clean_key(self, key, equal=lambda x, y: x - 1e-6 < y < x + 1e-6):
        """set similar key values all to be `key`, return `key`.

        Use method `set_clean` to access and change the clean-key
        dictionary *value* more conveniently.
        """
        meta_data = self.pop('meta_data', None)
        while self._near_key(key, equal) is not key:
            k = self._near_key(key, equal)
            assert k != key
            self[key] += self[k]  # def join_(a, b): return list(a) + list(b) could become input
            del self[k]  # del a superfluous reference
        if meta_data is not None:
            self['meta_data'] = meta_data
        return key

    def get_near(self, key, equal=lambda x, y: x - 1e-6 < y < x + 1e-6):
        """get the merged values list of all nearby keys.

        Caveat: the returned value is a new list

        :See also: `clean`, `set_clean`.
        """
        res = [] if key not in self else list(self[key])  # prevent adding the key
        done_keys = [key]
        while self._near_key(key, equal, done_keys) is not key:
            k = self._near_key(key, equal, done_keys)
            done_keys += [k]
            res += self[k]
        return res

    def set_clean(self, key):
        """join all entries with similar `key` and return the new value,
        a joined list of all respective values.

        This is the same as `clean_key` which however returns the new key, not
        the values.

        Example::

            data.set_clean(key) += [new_data_point]

            # same as
            data[data.clean_key(key)] += [new_data_point]

            # or more explicite, however with a different order of the data
            data[key] += [new_data_point]
            data.clean_key(key)  # joins data, however in the "wrong" order

            # similar as
            data[key] += [new_data_point]
            data.clean()  # cleans *all* keys

        """
        self.clean_key(key)
        return self[key]  # same as self[self.clean_key(key)]

    def _near_key(self, key, equal=lambda x, y: x - 1e-6 < y < x + 1e-6,
                  exclude=None):
        """return a key in self which is ``equal`` to ``key`` and otherwise ``key``.
        """
        if exclude is None:
            exclude = []
        for k in sorted(self.keys()):  # sorted doesn't work with float and str
            try:  # prevent type error from equal(float, str)
                if equal(k, key) and k != key and k not in exclude:
                    return k
            except TypeError:
                pass
        return key

    @property
    def successes(self):
        """return a class instance with attributes `x` (i.e. keys), `n`,
        `nsucc`, and `rate` as arrays.

        TODO: consider using cma.utilities.utils.DictClass2 instead namedtuple?
        """
        keys = [k for k in sorted(self) if np.isscalar(k)]
        nsucc = np.asarray([sum(np.isfinite(self[k])) for k in keys])
        n = np.asarray([len(self[k]) for k in keys])
        try:
            Successes = namedtuple('Successes',
                                   ['x', 'nsucc', 'n', 'rate'])
            res = Successes(np.asarray(keys), nsucc, n, nsucc / n)
        except:
            res = ClassFromDict(
                {'x': np.asarray(keys),
                 'nsucc': nsucc,
                 'n': n,
                 'rate': nsucc / n,
                 })
        return res

    def percentile(self, prctile, agg=sp1, samples=100):
        """TODO:review percentile based on bootstrapping"""
        raise NotImplementedError
        keys = [k for k in sorted(self) if np.isscalar(k)]
        res = []
        for k in keys:
            bstrapped = []
            for i in range(samples):
                idx = np.random.randint(len(self[k]))
                data = self[k][idx]
                bstrapped.append(agg(data))
            res.append(np.percentile(bstrapped, prctile))
        return np.asarray(keys), np.asarray(res)

    def __repr__(self):
        return repr(dict(self))

class ResultsPandas:
    """WIP ad hoc code for writing (changing) results in a `pandas.DataFrame` and to disk,

    where "results" refers to end-results of experiment repetitions rather
    than the traces of single runs.

    Caveat: this is an ad hoc implementation, some interfaces may be incomplete,
    interface details are still versatile and in flux, some recent changes may be
    faulty.

    Main features, similar to `Results`, are

    * Intermediate savings of results which consequently can be loaded from a
      different shell while the experiment is running.
    * Backup under the current timestamp before each saving into a
      ``'backups-name'`` folder (optional but default).
    * Similar float values can be "equalized" for correct data aggregation, see
      the `check_close_values` and `equalize_close_values` methods. This uses
      `np.isclose` which considers 1e-8 to be close to 1e-9 and 1+1e-5 close to
      1+1e-6.

    Compared to `Results`, this class is useful to store more information for
    each run, like the termination condition or a final condition number or
    constraints violations or meta parameter information. (In contrast, with
    `Results` the workaround for catching the final condition would writing a
    nonfinite entry when the target was not reached.)

    A guiding code example::

        import cma.experimentation

        res = cma.experimentation.ResultsPandas('some-name')  # reloads data to append/continue
        for dim in dimensions:
            es = cma.CMA(dim * [2], 1, {'verbose': -9})
            # a tracker in case we want to track, say, a minimum over the trace as result
            es.optimize(cma.ff.rosen, callback=my_result_tracker)
            if es.opts.get('verbose') == -9:  # do not save data while testing the setup
                res.append([es.N, es.popsize,
                            es.result.evaluations,
                            1 if 'ftarget' in es.stop() else 0,
                            repr(es.stop),
                            es.condition_number,
                            my_result_tracker.my_val_of_interest],
                            colums=['dimension', 'popsize',
                                    'evaluations',
                                    'targethit',
                                    'stopcondition',
                                    'conditionnumber',
                                    'myvalue'])
                res.save()

        # load the data:
        res = cma.experimentation.ResultsPandas('some-name')
        print(res.summary)  # summary statistics of columns in a dict
        res.df  # the pandas data frame

    Notes: ``self.df.drop(...)`` could be used to return a data frame with some
    entried dropped.
"""
    def __init__(self, filepathname, format='.csv'):
        """load data when `filepathname` exists.

        ``format='.csv'`` is human readible, however ``'.feather'`` is much more
        performant, ``'.parquet'`` should work too.
        """
        try:
            import pandas as pd  # noqa: F811
        except ImportError:
            warnings.warn("Please 'pip install pandas' to use the `ResultsPandas` class")
            raise

        self.name = filepathname
        self._extension = format  # feather format seems best performing binary 
                                  # see https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d
        # was: self.backup_dirname = '._{0}_backups'.format(self.name)
        if format == '.feather':
            try:
                import pyarrow
            except ImportError:
                warnings.warn("to use the .feather format: 'pip install pyarrow'"
                            '\nusing .csv for the time being')
                self._extension = '.csv'
        p, n = os.path.split(self.name)
        self.backup_dirname = os.path.join(p, 'backups-' + n)
        if os.path.exists(self.name + self._extension):
            self.load()
        else:
            self.df = pd.DataFrame()
        warnings.warn("The ResultsPandas class is versatile and has not be thoroughly tested",
                      category=_warnings_and_exceptions.NeverTestedWarning)
    def __getattr__(self, name):
        """access the `DataFrame` ``Results.df`` directly from `Results`"""
        return getattr(self.df, name)

    @property
    def summary(self):
        """return number of finite entries, number of different values, and

        (min, max) for each column.
        """
        res = self
        samples = {}
        for name in res.df.columns:
            min_, max_ = res.df[name].dropna().min(), res.df[name].dropna().max()
            if isinstance(min_, (int, np.integer)) and isinstance(max_, (int, np.integer)):
                min_, max_ = int(min_), int(max_)
            else:
                try:
                    min_, max_ = float(min_), float(max_)
                except Exception:
                    pass
            try:
                n_valid = len(res.df[name].dropna())
            except Exception:
                n_valid = int(sum(v not in ('nan', '') for v in res.df[name]))
            samples[name] = [n_valid,
                             len(set(res.df[name])),
                              (min_, max_)]
        return samples

    def save(self, backup=True, time_s=2, **kwargs):
        """save data, warn when saving takes more than `time_s` seconds.

        `kwargs` are passed to the saving method of the `pandas` data frame.

        Details: `save` is in essence just a shortcut for::

            self.backup()
            self.df.to_feather(self.name + self.extension)  # assuming .extension == '.feather'

        TODO: generalize by passing the `DataFrame` method name for saving, like
        ``.save('to_feather')``? Annoyingly, pandas does not add a proper
        extension by default.
        """
        backup and self.backup()  # TODO: saves wrong format when self._extension changed
        try:
            os.makedirs(os.path.dirname(self.name + self._extension))  # bails when a folder exists
        except Exception:  # error or IOError or OSError or FileExistsError?
            pass
        with TimeWarner('save', time_s):
            if self._extension == '.csv':
                kwargs.setdefault('index', False)
                self.df.to_csv(self.name + self._extension, **kwargs)
            elif self._extension == '.parquet':
                self.df.to_parquet(self.name + self._extension, **kwargs)
            else:
                if self._extension != '.feather':
                    warnings.warn("'{0}' extension not recognized, saving in feather format"
                                  .format(self._extension))
                self.df.to_feather(self.name + '.feather', **kwargs)
                # see https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d
    def load(self, polish=False):
        import pandas as pd  # noqa: F811
        with TimeWarner('ResultsPandas.load'):
            if self._extension == '.csv':
                self.df = pd.read_csv(self.name + self._extension)
            elif self._extention == '.parqet':
                self.df = pd.read_parquet(self.name + self._extension)
            else:
                self.df = pd.read_feather(self.name + self._extension)
        # self.equalize_close_values()
        # polish and self._polish()
        # self.check()
        return self
    def equalize_close_values(self, column):
        df = self.df
        s = sorted(set(df[column]))
        for i, t in enumerate(np.isclose(s[:-1], s[1:])):
            if t:  # values s[i] and s[i+1] are (too) close -> copy s[i] value to rows with s[i+1] value
                df.loc[getattr(df, column) == s[i+1], column] = s[i]
                # df.loc[...] is a view df[...] is a copy    
    def _polish(self, column):
        """a quick hack"""
        self.df['success'] = ['callback' in s for s in self.df.stop]  # slower: self.df.stop.apply(lambda s: 'callback' in s)
        # insert best column (best by dimension)
        idx = self.df_succ.groupby(['dimension']).evals.idxmin()  # don't know how to make .median() work
        self.df['best_' + column] = [getattr(self.df.loc[idx[dim]], column) if dim in idx else np.nan
                                   for dim in self.df.dimension]
    def check_close_values(self, column):
        s = sorted(set(getattr(self.df, column)))
        if any(np.isclose(s[:-1], s[1:])):
            warnings.warn("Found ambiguous double values in column {0} in {1}."
                          "Call .equalize_close_values() and .save() to fix."
                          .format(column, s))

    def append(self, data, columns, kwargs_df={}, kwargs_concat={}):
        """append a single data row"""
        # self.df = self.df.append(*args, **kwargs)
        self.extend([data], columns, kwargs_df, kwargs_concat)
    def extend(self, data, columns, kwargs_df={}, kwargs_concat={}):
        """extend frame by data which is a sequence of rows"""
        import pandas as pd  # noqa: F811

        kwargs_concat.setdefault('ignore_index', True)
        self.df = pd.concat([self.df,
            pd.DataFrame.from_records(data, columns=columns, **kwargs_df)], **kwargs_concat)
    def drop(self, *args, **kwargs):
        """call ``self.df.drop`` and reassign result"""
        raise NotImplementedError
        self.df = self.df.drop(*args, **kwargs)
    def reset_index(self):
        """caveat: the original index is lost which may be undesirable"""
        self.df.reset_index(drop=True, inplace=True)
    def column_values(self, column, by='dimension'):
        """a sorted list of values in the column of name `column`."""
        df, res = self.df, {}
        if by:
            for dimension in sorted(set(df[by])):
                res[dimension] = sorted(set(df[df[by] == dimension][column]))
        else:
            res = sorted(set(df[column]))
        return res              
    @property
    def failure_indices(self, column):
        """TODO: revise such that we have a boolean failed_setting column"""
        failed = [d.Index for d in self.df.itertuples() if 'callback' not in d.stop]
        def failed_parameter(d, failed_indices):
            for i in failed_indices:
                if (d.dimension, getattr(d, column)) == (self.df.loc[i].dimension, getattr(self.df.loc[i], column)):
                    return True
            return False
        return [d.Index for d in self.df.itertuples() if failed_parameter(d, failed)]
    def print_checks(self, column):
        """do not use `iterrows` but `itertuples`
        https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
        """
        df = self.df
        print("{0} data, {1} failures, failure_indices = {2}\n".format(len(df), len(self.failure_indices), self.failure_indices),
              "min {0} fraction = {1:.4}".format(column, min(np.exp(np.diff(np.log(sorted(set(df[column]))))))),
             )

    def backup(self):
        """backup saved data by making a file copy"""
        dest = os.path.join(self.backup_dirname,
                            unique_time_stamp(2) + self._extension)
        copy_file(self.name + self._extension, dest)
        self.last_backup = dest

del division, print_function
