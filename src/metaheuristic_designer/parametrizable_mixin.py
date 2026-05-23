"""
Module providing the :class:`ParametrizableMixin` for managing schedulable parameters.
"""

import logging

logger = logging.getLogger(__name__)


class ParametrizableMixin:
    """Mixin that turns raw keyword arguments into dynamic, schedulable parameters.

    Any argument passed to :meth:`store_kwargs` or :meth:`update_kwargs`
    can be a plain value or a *callable* that receives the current
    progress (a float between 0 and 1) and returns the value to use.
    The current values are accessible via :meth:`get_params` or the
    :attr:`params` property.
    """

    def __init__(self):
        super().__init__()
        self.raw_kwargs = {}
        self.current_kwargs = {}

    def store_kwargs(self, progress: float = 0, **kwargs):
        """Store keyword arguments and evaluate them at the given progress.

        Parameters
        ----------
        progress : float, optional
            Progress value forwarded to any callable parameters.
        **kwargs
            Parameter names and values (constants or callables).
        """

        self.raw_kwargs.update(kwargs)
        self.update(progress=progress)

    def update(self, progress: float):
        """Re-evaluate all stored parameters at the current progress.

        Parameters
        ----------
        progress : float
            Current progress (0-1) used to evaluate callable parameters.
        """

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
        """Add or replace parameters and immediately evaluate them.

        Parameters
        ----------
        progress : float, optional
            Progress value forwarded to any callable parameters.
        **kwargs
            Parameter names and new values.
        """

        self.raw_kwargs.update(kwargs)
        for k, v in kwargs.items():
            if callable(v):
                self.current_kwargs[k] = v(progress)
            else:
                self.current_kwargs[k] = v

    def get_params(self) -> dict:
        """Return a copy of the current parameter dictionary.

        Returns
        -------
        dict
            The evaluated (non-callable) parameter values.
        """

        return self.current_kwargs.copy()

    @property
    def params(self):
        """Access parameter values by attribute-style lookup.

        Returns
        -------
        _ParamView
            A helper object that reads from the current parameter
            dictionary.
        """

        return _ParamView(self)


class _ParamView:
    """Lightweight wrapper that exposes a dictionary as attributes.

    ``obj.params.xxx`` is equivalent to ``obj.get_params()["xxx"]``,
    raises ``AttributeError`` when the key is missing.
    """

    def __init__(self, owner):
        self._owner = owner

    def __getattr__(self, name):
        try:
            return self._owner.current_kwargs[name]
        except KeyError:
            raise AttributeError(name)
