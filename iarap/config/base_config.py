from types import ModuleType
from typing import Tuple, Type, Any
from dataclasses import dataclass


# Pretty printing class
class PrintableConfig:
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)
    

@dataclass
class FactoryConfig(PrintableConfig):
    """Config class for instantiating objects given their class name and module."""

    _name: str
    _module: ModuleType

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        target = getattr(self._module, self._name)
        add_kwargs = vars(self).copy()
        add_kwargs.pop("_module")
        add_kwargs.pop("_name")
        add_kwargs.update(kwargs)
        return target(**add_kwargs)