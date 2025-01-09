__version__ = "0.0.4"

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