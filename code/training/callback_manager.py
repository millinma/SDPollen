
from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .training import ModularTaskTrainer


CALLBACK_FUNCTIONS = [
    "cb_on_train_begin",
    "cb_on_train_end",
    "cb_on_iteration_begin",
    "cb_on_iteration_end",
    "cb_on_loader_exhausted",
    "cb_on_step_begin",
    "cb_on_step_end",
]

CALLBACK_MODULES = [
    "data",
    "model",
    "optimizer",
    "scheduler",
    "criterion",
    "curriculum",
]


class CallbackSignature:
    @abstractmethod
    def cb_on_train_begin(
        self,
        trainer: ModularTaskTrainer
    ) -> None:
        pass

    @abstractmethod
    def cb_on_train_end(
            self,
            trainer: ModularTaskTrainer,
            test_results: dict
    ) -> None:
        pass

    @abstractmethod
    def cb_on_iteration_begin(
        self,
        trainer: ModularTaskTrainer,
        iteration: int
    ) -> None:
        pass

    @abstractmethod
    def cb_on_iteration_end(
        self,
        trainer: ModularTaskTrainer,
        iteration: int,
        metrics: dict
    ) -> None:
        pass

    @abstractmethod
    def cb_on_loader_exhausted(
        self,
        trainer: ModularTaskTrainer,
        iteration: int
    ) -> None:
        pass

    @abstractmethod
    def cb_on_step_begin(
        self,
        trainer: ModularTaskTrainer,
        iteration: int,
        batch_idx: int
    ) -> None:
        pass

    @abstractmethod
    def cb_on_step_end(
        self,
        trainer: ModularTaskTrainer,
        iteration: int,
        batch_idx: int,
        loss: float
    ) -> None:
        pass


class CallbackManager:
    def __init__(self, trainer: ModularTaskTrainer):
        self.callbacks = {cb: [] for cb in CALLBACK_FUNCTIONS}
        self._register(trainer)

    def callback(self, position: str, **kwargs) -> None:
        if position not in self.callbacks:
            raise ValueError(f"Callback position {position} not found.")
        for cb in self.callbacks[position]:
            cb(**kwargs)

    def _register(self, trainer: ModularTaskTrainer) -> None:
        for module in CALLBACK_MODULES:
            for callback_name in CALLBACK_FUNCTIONS:
                m = getattr(trainer, module, None)
                if m is None:
                    continue
                cb = getattr(m, callback_name, None)
                if cb is None:
                    continue
                self._check_signature(m, cb, callback_name)
                self.callbacks[callback_name].append(cb)

    def _check_signature(
        self,
        module: object,
        func: callable,
        callback_name: str
    ) -> None:
        template_method = getattr(CallbackSignature, callback_name, None)
        if template_method is None:
            raise ValueError(
                f"Callback {callback_name} not found in CallbackSignature."
            )
        func_sig = inspect.signature(func)
        template_sig = inspect.signature(template_method)
        template_params = list(template_sig.parameters.values())[1:]
        modified_template_sig = template_sig.replace(
            parameters=template_params)
        if not func_sig == modified_template_sig:
            raise TypeError(
                f"Callback {callback_name} in {module.__class__.__name__}"
                " does not match the expected signature.\n"
                f"\tGot: {func_sig}\n"
                f"\tExpected: {modified_template_sig}"
            )
