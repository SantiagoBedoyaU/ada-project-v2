from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from strategies.estrategia_1 import solve as solve1
from strategies.estrategia_2 import solve as solve2
from strategies.estrategia_3 import solve as solve3

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
    strategy: Annotated[str, Form()],
    initial_state: Annotated[str, Form()],
    candidate_system: Annotated[str, Form()],
    present_subsystem: Annotated[str, Form()],
    future_subsystem: Annotated[str, Form()],
):
    strategy = int(strategy)
    if strategy == 1:
        [min_emd_key, min_emd_result, time] = await solve1(tpms, initial_state, candidate_system, present_subsystem, future_subsystem)
    elif strategy == 2:
        [min_emd_key, min_emd_result, time] = await solve1(tpms, initial_state, candidate_system, present_subsystem, future_subsystem)
    else:
        [min_emd_key, min_emd_result, time] = await solve3(tpms, initial_state, candidate_system, present_subsystem, future_subsystem)
    return {
        "min_emd_key": min_emd_key,
        "min_emd_result": min_emd_result,
        "time": time,
    }