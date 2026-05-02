from .silent_reporter import SilentReporter

def create_reporter(name, **kwargs):
    if name == "silent":
        return SilentReporter()