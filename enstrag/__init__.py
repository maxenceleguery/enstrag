__version__ = "0.1.4"

import os

def verify_execution() -> None:
    import socket

    class ExecutionOnFrontal(Exception):
        pass

    hostname = socket.gethostname().split(".")[0]

    if hostname.startswith("mesogip"):
        raise ExecutionOnFrontal("Please don't run code on the frontal. Use srun to start an instance.")
    elif hostname.startswith("ensta"):
        print(f"Starting RAG on {hostname}")
    else:
        print(f"Starting RAG on unknown {hostname}")

def get_args():
    import argparse
    parser = argparse.ArgumentParser(prog='Enstrag')

    parser.add_argument('-r', '--reset', action='store_true', help="Reset the vector database on start.")
    parser.add_argument('-v', '--version', action='store_true', help="Show version")
    #parser.add_argument('-x', '--explained', action='store_true', help="Provide explaination (XAI)")
    parser.add_argument('-l', '--local', action='store_true', help="Deactivate Gradio share feature")
    parser.add_argument('-s', '--server', action='store_true', help="Enable FastAPI server mode")
    parser.add_argument('--persist_dir', type=str, default="/home/ensta/ensta-leguery/enstrag_folder")
    parser.add_argument('--models_path', type=str, default="/home/ensta/data")
    parser.add_argument('--llm_folder', type=str, default="Qwen2.5-1.5B-Instruct")
    parser.add_argument('--embedding_folder', type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    if args.version:
        from enstrag import __version__
        print(__version__)
        exit(0)
    
    os.environ["MODELS_PATH"] = args.models_path
    os.environ["PERSIST_PATH"] = args.persist_dir

    return args