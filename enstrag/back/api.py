from fastapi import FastAPI

def build_server(agent):
    app: FastAPI = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "Hello World"}
    
    return app