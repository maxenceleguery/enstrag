import os
import gradio as gr

from .base_front import Front
from .gradio_component import build_admin_panel, build_qa_panel

class GradioFront(Front):
    def launch(self, **kwargs) -> None:
        rag = build_qa_panel(self.agent)
        admin = build_admin_panel(self.agent)

        demo = gr.TabbedInterface([rag, admin], tab_names=["Enstrag Bot", "Admin panel"])

        demo.launch(
            allowed_paths=[os.environ.get("PERSIST_PATH")],
            **kwargs,
        )