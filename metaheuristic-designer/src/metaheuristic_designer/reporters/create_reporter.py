from .tqdm_reporter import TQDMReporter
from .verbose_reporter import VerboseReporter
from .silent_reporter import SilentReporter


def create_reporter(reporter_name: str, **kwargs):
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
