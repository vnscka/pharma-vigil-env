from fastapi import FastAPI

app = FastAPI(title="pharma_vigil_env")

@app.get("/health")
def health():
    return {"status": "ok"}