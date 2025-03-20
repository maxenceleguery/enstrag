from .agent_client import AgentClient
from .gradio_front import GradioFront

agent = AgentClient()

front = GradioFront(agent)
front.launch(
    share=False,
    server_port=7861,
    ssl_keyfile="/app/https/key.pem", 
    ssl_certfile="/app/https/cert.pem",
    ssl_verify=False,
)