from typing import Any
from enum import Enum
from uuid import UUID
from dataclasses import is_dataclass, asdict
from json import JSONEncoder


class DataJSONEncoder(JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, UUID):
            return str(obj)
        if is_dataclass(obj):
            return asdict(obj)
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return super().default(obj)
