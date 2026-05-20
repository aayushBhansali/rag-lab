from src.rag.data_loader.AbstractDataLoader import AbstractDataLoader
from pathlib import Path


class TextFileLoader(AbstractDataLoader):
    """
    Data loader for text files. Parses text files and returns a list of chunks.
    """
    def load(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        
        with open(path, 'r', encoding='utf-8') as fs:
            lines = fs.readlines()

        return lines
