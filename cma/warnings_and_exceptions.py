import warnings

max_warnings = {}
'''overwrites the default number of max warnings by warning name'''
skip_warnings = {}
'''overwrites the default number of skipped first warnings by warning name'''

def filterwarning(*args, **kwargs):
    """add warning filter but only if not already there (i.e. defined by the user),

    not clear why this works as expected.
    """
    try:
        _is_filtered = any(w[2] == kwargs['category'] for w in warnings.filters)
    except Exception:
        _is_filtered = False
    if not _is_filtered:
        warnings.filterwarnings(*args, **kwargs)

class NeverTestedWarning(UserWarning):
    """Feature has been implemented but never or poorly tested for correct results"""

class CountWarnings(object):
    """Count warnings to improve warning control.

    Calling this instance, see `__call__`, counts how often the passed object
    instance was called with the given warning name and returns `True` iff
    ``skip_warns < count <= max_warns``.

    The ``max_warns`` and ``skip_warns`` parameters can be modified by the user
    for each "warning name" by changing the module variables
    ``max_warnings:dict`` and ``skip_warnings:dict``, respectively, for example
    like ``cma.warnings_and_exceptions.max_warnings['warning name'] = 3``.

    Usage for developers::

        from cma import warnings_and_exceptions as _we

        if _we.deliver_warning(self, 'new warning name', max_warns=3):
            warnings.warn('my message. ' + 
                          _we.deliver_warning.message())

    `warnings.warn` is called in place to not obscure the code line to which the
    warning refers to.

    The internal representation is by a dictionary like
    ``.counters[(object_instance, warning_name)] == [warnings_done, max_warns,
    skip_warns]``.
    """
    def __init__(self):
        self.counters = {}
    def message(self, message="", object_instance=None, warning_name=None):
        """append count to `message` and the warning name to allow to change max_warns"""
        if object_instance is None:
            object_instance = self._last_key[0]
        if warning_name is None:
            warning_name = self._last_key[1]
        try:
            d = self.counters[(object_instance, warning_name)] 
        except Exception as e:
            warnings.warn("Caught exception {0}"
                          "\n  in CountWarnings.message"
                          "\n  Please consider to report this message at"
                          "\n  https://github.com/CMA-ES/pycma/issues"
                          .format(e))
            return message
        s = ' Warning "{2}":{0}/{1}.'.format(d[0], d[1] + d[2], warning_name)
        if d[0] == d[1] + d[2]:
            s = s + " Further warnings are surpressed."
        return message + s  # TODO: review, is there a better way?
    def __call__(self, object_instance, warning_name, max_warns=1, skip_warns=0):
        """increment warning count and return ``0 < count - skip_warns <= max_warns``

        which is by default `True` for the first warning and `False` otherwise.
        `skip_warns` indicates how many initial warnings shall be ignored and
        `max_warns` indicates how many warnings should be exposed to the user.

        The warning is identified by ``(object_instance, warning_name)``, such
        that the warnings count starts with each instance anew.

        `object_instance` and `warning_name` must be hashable objects which in
        combination give a unique ID for the warnings count.

        Details: having two id input arguments is somewhat arbitrary and
        ``deliver_warning(0, (self, name))`` behaves identical to
        ``deliver_warning(self, name)``. However, the user control of
        ``max_warn` via `cma.warnings_and_exceptions.max_warnings`` should and
        does not depend on the specific instance, i.e. the first of the two id
        inputs.
        """
        max_warns = max_warnings.get(warning_name, max_warns)
        skip_warns = skip_warnings.get(warning_name, skip_warns)
        try:
            if (object_instance, warning_name) not in self.counters:
                self.counters[(object_instance, warning_name)] = [0, max_warns, skip_warns]
        except TypeError as e:
            warnings.warn("`CountWarnings` control can not be applied on this instance"
                          " because of the exception\n    {0}\n"
                          "Most likely, ``({1}, {2})`` is not a valid `dict` key, because {1}"
                          " is not hashible. \nself={3}"
                          "\n  Please consider to report this problem at"
                          "\n  https://github.com/CMA-ES/pycma/issues"
                          .format(e, object_instance, warning_name, self.__dict__))
            return True
        self._last_key = object_instance, warning_name
        self.counters[self._last_key][0] += 1
        return 0 < self.counters[self._last_key][0] - skip_warns <= max_warns

deliver_warning = CountWarnings()
