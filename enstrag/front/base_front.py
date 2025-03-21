import abc

class Front(abc.ABC):
    def __init__(self, agent):
        self.agent = agent

    @abc.abstractmethod
    def launch(self, share: bool = False) -> None:
        ...