import json

class Config:
    def __init__(self, path) -> None:
        self.path = path

        with open(self.path, mode='r', encoding='utf-8') as f:
            self.data = json.load(f)
            self.__dict__.update(self.data)

    @property
    def dict(self):
        return self.__dict__
        