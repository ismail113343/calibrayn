from pathlib import Path
from string import Template


class PromptBook:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._cache = {}

    def render(self, name, /, **fields):
        if name not in self._cache:
            path = self.directory / f"{name}.txt"
            if not path.is_file():
                raise FileNotFoundError(f"Prompt file not found: {path}")
            system, divider, user = path.read_text(encoding="utf-8").partition("\n---\n")
            if not divider:
                raise ValueError(f"{path.name} needs a line with --- between its two parts")
            self._cache[name] = (Template(system.strip()), Template(user.strip()))
        system, user = self._cache[name]
        return system.substitute(fields), user.substitute(fields)
