import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
from app.consumer import run_in_background
from app.cleanup import start_cleanup_loop

def create_app():
    app = FastAPI(title="Sentinel AI Result API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app

app = create_app()

@app.on_event("startup")
def _startup():
    run_in_background()
    start_cleanup_loop()

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
