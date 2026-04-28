import pytest
from metaheuristic_designer.parametrizable_mixin import ParametrizableMixin
from metaheuristic_designer.parameter_schedules.linear_schedule import LinearSchedule
class DummyComponent(ParametrizableMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.store_kwargs(**kwargs)


def test_store_static_kwargs():
    comp = DummyComponent(scale=0.5, name="test")
    assert comp.get_params() == {"scale": 0.5, "name": "test"}


def test_store_callable_kwarg_evaluated_immediately():
    comp = DummyComponent(scale=lambda p: p * 10)
    params = comp.get_params()
    assert params["scale"] == 0.0   # progress 0 → 0


def test_step_updates_callable():
    comp = DummyComponent(scale=lambda p: p * 10)
    comp.step(progress=0.3)
    assert comp.get_params()["scale"] == 3.0


def test_step_leaves_static_unchanged():
    comp = DummyComponent(scale=1.0)
    comp.step(progress=0.7)
    assert comp.get_params()["scale"] == 1.0


def test_get_params_returns_copy():
    comp = DummyComponent(scale=2.0)
    params = comp.get_params()
    params["scale"] = 999
    assert comp.get_params()["scale"] == 2.0   # original untouched


def test_schedule_object_as_callable():
    """A SchedulableParameter subclass should work because it's callable."""
    comp = DummyComponent(amount=LinearSchedule(init_value=0, final_value=100))
    comp.step(progress=0.5)
    assert comp.get_params()["amount"] == 50.0


def test_multiple_mixed_params():
    comp = DummyComponent(
        static=42,
        dynamic=lambda p: 100 - p * 100
    )
    comp.step(progress=0.2)
    params = comp.get_params()
    assert params["static"] == 42
    assert params["dynamic"] == 80.0


def test_step_called_multiple_times():
    comp = DummyComponent(value=lambda p: 2.0 ** p)
    for p in [0.0, 0.5, 1.0]:
        comp.step(progress=p)
    assert comp.get_params()["value"] == 2.0