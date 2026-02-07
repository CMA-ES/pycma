import warnings

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
