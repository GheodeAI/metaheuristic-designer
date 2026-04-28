import logging

logger = logging.getLogger(__name__)


class ParametrizableMixin:
    def __init__(self):
        super().__init__()
        self.raw_kwargs = {}
        self.current_kwargs = {}

    def store_kwargs(self, progress: float = 0, **kwargs):
        self.raw_kwargs.update(kwargs)
        self.step(progress=progress)

    def step(self, progress: float):
        for k, v in self.raw_kwargs.items():
            if callable(v):
                self.current_kwargs[k] = v(progress)
            else:
                self.current_kwargs[k] = v

        if logger.isEnabledFor(logging.DEBUG):
            if hasattr(self, "name"):
                name = self.name
            else:
                name = "unknown"
            cls_name = self.__class__.__name__
            logger.debug('Evaluated every parameter at %.4f for %s named "%s".', progress, cls_name, name)

    def update_kwargs(self, progress: float = 0, **kwargs):
        self.raw_kwargs.update(kwargs)
        for k, v in kwargs.items():
            if callable(v):
                self.current_kwargs[k] = v(progress)
            else:
                self.current_kwargs[k] = v

    def get_params(self) -> dict:
        return self.current_kwargs.copy()

    @property
    def params(self):
        return _ParamView(self)


class _ParamView:
    def __init__(self, owner):
        self._owner = owner

    def __getattr__(self, name):
        try:
            return self._owner.current_kwargs[name]
        except KeyError:
            raise AttributeError(name)
