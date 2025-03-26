from .agent_client import AgentClient
from .gradio_front import GradioFront

if __name__ == '__main__':
    agent = AgentClient()
    front = GradioFront(agent)
    front.launch(share=False, server_port=7860)
