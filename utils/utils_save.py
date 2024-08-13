import json
from typing import Any, Dict, List

import numpy as np

from worldllm_envs.envs.base import BaseRule


def to_json_serializable(obj: Any) -> Any:
    """Convert an object to a JSON serializable object"""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, BaseRule):
        return obj.get_prompt()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")


class RuleOutput:
    """Data format to save rules and scores"""

    def __init__(
        self,
        true_rule: BaseRule,
        rules: List[str],
        likelihoods: List[float],
        metrics: Dict[str, Any],
    ) -> None:
        self.true_rule = true_rule
        self.rules = rules
        self.likelihoods = likelihoods
        self.metrics = metrics

    def to_json(self, filename):
        data = {
            "true_rule": self.true_rule,
            "rules": self.rules,
            "likelihoods": self.likelihoods,
            "metrics": self.metrics,
        }
        data_serialized = to_json_serializable(data)
        with open(filename, "w") as f:
            json.dump(data_serialized, f)
