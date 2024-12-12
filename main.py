from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from estrategia_1 import solve
import time

app = FastAPI()
origins = ["*"]
app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

@app.post("/solve")
async def solve_problem(
    tpms: Annotated[list[UploadFile], File()],
    initial_state: Annotated[str, Form()],
    candidate_system: Annotated[str, Form()],
    present_subsystem: Annotated[str, Form()],
    future_subsystem: Annotated[str, Form()],
):
    inicio = time.perf_counter()
    [min_emd_key, min_emd_result] = await solve(tpms, initial_state, candidate_system, present_subsystem, future_subsystem)
    fin = time.perf_counter()
    return {
        "min_emd_key": min_emd_key,
        "min_emd_result": min_emd_result,
        "time": fin-inicio,
    }