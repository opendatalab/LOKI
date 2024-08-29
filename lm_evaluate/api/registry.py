from lm_evaluate.api.model import LMM
from lm_evaluate.api.task import Task 

from loguru import logger as eval_logger

MODEL_REGISTRY = {}
TASK_REGISTRY = {}


def register_model(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(cls, LMM), f"Model '{name}' ({cls.__name__}) must extend LMM class"

            assert name not in MODEL_REGISTRY, f"Model named '{name}' conflicts with existing model! Please register with a non-conflicting alias instead."

            MODEL_REGISTRY[name] = cls
        return cls

    return decorate


def register_task(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(cls, Task), f"Task '{name}' ({cls.__name__}) must extend Task class"

            assert name not in TASK_REGISTRY, f"Task named '{name}' conflicts with existing task! Please register with a non-conflicting alias instead."

            TASK_REGISTRY[name] = cls
        return cls

    return decorate