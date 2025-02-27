import abc

from ..rag import RagAgent

class Front(abc.ABC):
    def __init__(self, agent: RagAgent):
        self.agent = agent

    @abc.abstractmethod
    def launch(self, share: bool = False) -> None:
        ...