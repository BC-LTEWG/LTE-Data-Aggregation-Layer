import os
from .parameters import Params
from .Overseer import Overseer
from typing import Tuple
import numpy as np
import subprocess
import json
from scipy.linalg import inv

# REPLACE WITH THE FILE PATH TO YOUR OWN BINARY. IF USING WINDOWS, MAKE SURE YOU USE DOUBLE FORWARD SLASHES (e.g. C:\\Users\\...)
EXE_PATH = "/home/lennyyyyyyyy/Github/Labor-Time-Economy-Simulation/bin/sim"
LOG_PATH = "/home/lennyyyyyyyy/Documents/Overseer/logs/log.jsonl"

def get_trajectories(params: Params):
    overseer = Overseer(EXE_PATH, params, LOG_PATH)
    sim_finished = False
    while not sim_finished:
        sim_finished = overseer.step()
        traj = overseer.get_data()
        yield traj
