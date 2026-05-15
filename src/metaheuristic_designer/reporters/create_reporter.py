"""
Factory function for reporter objects.
"""

from ..reporter import Reporter
from .tqdm_reporter import TQDMReporter
from .verbose_reporter import VerboseReporter
from .silent_reporter import SilentReporter


def create_reporter(reporter_name: str, **kwargs) -> Reporter:
    """Instantiate a reporter by name.

    Parameters
    ----------
    reporter_name : str
        One of ``"silent"``, ``"tqdm"``, or ``"verbose"``.
    **kwargs
        Forwarded to the reporter constructor.

    Returns
    -------
    Reporter
        A concrete reporter instance.

    Raises
    ------
    ValueError
        If *reporter_name* is not recognised.
    """

    reporter = None
    match reporter_name:
        case "silent" | "nothing":
            reporter = SilentReporter(**kwargs)
        case "tqdm":
            reporter = TQDMReporter(**kwargs)
        case "verbose":
            reporter = VerboseReporter(**kwargs)
        case _:
            raise ValueError(f'Reporter type "{reporter_name}"not available.')

    return reporter
