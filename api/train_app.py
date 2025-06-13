from fastapi import BackgroundTasks, FastAPI
import uuid, mlflow, os
from utils.train_pipeline import run_training

app = FastAPI()

@app.post("/train")
def train(bt: BackgroundTasks):
    run_id = str(uuid.uuid4())
    bt.add_task(run_training)       # lanza el Pipeline en segundo plano
    return {"status": "started", "run_id": run_id}
