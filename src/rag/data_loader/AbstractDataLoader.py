from abc import ABC, abstractmethod
from pathlib import Path

class AbstractDataLoader(ABC):

    @abstractmethod
    def load(self, path: Path) -> list[str]:
        pass
