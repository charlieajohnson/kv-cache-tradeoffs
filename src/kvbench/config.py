from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ExperimentConfig:
    data: Dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.data
        for key in keys:
            if not isinstance(node, dict):
                return default
            if key not in node:
                return default
            node = node[key]
        return node

    def merge(self, updates: Dict[str, Any]) -> "ExperimentConfig":
        def _merge(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
            out = copy.deepcopy(left)
            for k, v in right.items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = _merge(out[k], v)
                else:
                    out[k] = v
            return out

        return ExperimentConfig(_merge(self.data, updates))
