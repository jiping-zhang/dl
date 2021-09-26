from abc import abstractmethod
try:
    import dl.nn as nn
except ModuleNotFoundError as err:
    import nn

class Optimizer:
    def __init__(self, model: nn.Layer, lr: float) -> None:
        self._model = model
        self._lr = lr

    def step(self) -> None:
        self._model.step(self._lr)


class SGD(Optimizer):
    def __init__(self, model: nn.Layer, lr: float) -> None:
        super().__init__(model, lr)


class Momentum(Optimizer):
    def __init__(self, model: nn.Layer, lr: float, momentum_factor: float = 0.9) -> None:
        super().__init__(model, lr)
        self.__last_batch_grad = None
        self.__current_batch_grad = None
        self.__momentum_factor = momentum_factor

    def step(self) -> None:
        self.__current_batch_grad = self._model.get_self_gradient()
        if self.__last_batch_grad is not None:
            self._model.add_from_gradient(self.__momentum_factor*self.__last_batch_grad)
        self._model.step(self._lr)
        self.__last_batch_grad = self.__current_batch_grad


class Schedule:
    def __init__(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer
        self._initial_lr = optimizer._lr

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError


class ConnectableSchedule(Schedule):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


INFINITY = -1


class NormalSchedule(ConnectableSchedule):
    def __init__(self, optimizer: Optimizer, factor: float = 1.0, steps: int = INFINITY) -> None:
        super().__init__(optimizer)
        self.__factor = factor
        self.__steps = steps

    def __len__(self) -> int:
        return self.__steps

    def step(self) -> None:
        self._optimizer._lr = self.__factor * self._initial_lr


class ExpSchedule(ConnectableSchedule):
    def __init__(self, optimizer: Optimizer, decay_factor: float = 0.99, steps_per_decay: int = 1, final_decay_times: int = INFINITY) -> None:
        super().__init__(optimizer)
        self.__decay_factor = decay_factor
        self.__steps_per_decay = steps_per_decay
        self.__final_decay_times = final_decay_times
        self.__steps = 0
        self.__total_decay_times = 0
        self.__k = 1.0

    def step(self) -> None:
        self.__steps += 1
        if self.__total_decay_times < self.__final_decay_times and self.__steps >= self.__steps_per_decay:
            self.__k *= self.__decay_factor
            self.__total_decay_times += 1
            self.__steps = 0
            self._optimizer._lr = self._initial_lr * self.__k

    def __len__(self) -> int:
        if self.__final_decay_times == INFINITY:
            return INFINITY
        return self.__final_decay_times*self.__steps_per_decay
