from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from uuid import uuid4
import subprocess
import pathlib
import json
import time

"""
uvicorn api:app --reload --port 8000
"""

app = FastAPI()

JOBS_FILE = pathlib.Path("jobs.json")

# In-memory fallback
jobs = {}

class WakewordRequest(BaseModel):
    wakeword: str

def save_jobs():
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))

def run_pipeline(job_id: str, wakeword: str):
    jobs[job_id]["status"] = "running"
    save_jobs()

    try:
        cmd = [
            "uv", "run", "pipeline.py",
            "--wakeword", wakeword
        ]
        with open(f"{job_id}.log", "w") as logf:
            subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=True)

        jobs[job_id]["status"] = "completed"
    except subprocess.CalledProcessError:
        jobs[job_id]["status"] = "failed"
    finally:
        jobs[job_id]["finished_at"] = time.time()
        save_jobs()


@app.post("/wakewords/register")
def register_wakeword(req: WakewordRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    jobs[job_id] = {
        "id": job_id,
        "wakeword": req.wakeword,
        "status": "pending",
        "created_at": time.time(),
        "log_file": f"{job_id}.log"
    }
    save_jobs()

    background_tasks.add_task(run_pipeline, job_id, req.wakeword)
    return {"job_id": job_id, "status": "queued"}


@app.get("/wakewords/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    return jobs[job_id]
