from __future__ import annotations

from abc import ABC, abstractmethod

from ptx_gym.evaluation.types import EvaluatedCandidate, Payload
from ptx_gym.helpers.environment import resolve_git_commit_hash
from ptx_gym.kernels.base import TritonPTXKernel


class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, op) -> bool:
        raise NotImplementedError


class BaseCandidateEvaluator(ABC):
    def __init__(
        self,
        operator_cls: type[TritonPTXKernel],
        *,
        operator: TritonPTXKernel | None = None,
    ) -> None:
        self.operator_cls = operator_cls
        self.operator = operator_cls() if operator is None else operator
        self.kernel_name = operator_cls.__name__
        self.git_commit_hash = resolve_git_commit_hash()

    @abstractmethod
    def evaluate(
        self,
        payload: Payload,
    ) -> EvaluatedCandidate:
        raise NotImplementedError
