import os
import gradio as gr

from .base_front import Front
from .gradio_component import build_admin_panel, build_qa

class GradioFront(Front):
    def launch(self, share: bool = False, server_port: int = 7860) -> None:

        rag = build_qa(self.agent)
        admin_panel = build_admin_panel(self.agent)

        demo = gr.TabbedInterface([rag, admin_panel], tab_names=["Enstrag Bot", "Admin panel"])

        demo.launch(
            share=share,
            allowed_paths=[os.environ.get("PERSIST_PATH")],
            server_port=server_port,
        )