import os
from .parameters import Params
from .Overseer import Overseer
from typing import Tuple
import numpy as np
import subprocess
import json
from scipy.linalg import inv

# REPLACE WITH THE FILE PATH TO YOUR OWN BINARY. IF USING WINDOWS, MAKE SURE YOU USE DOUBLE FORWARD SLASHES (e.g. C:\\Users\\...)
EXE_PATH = "/home/alex/github/Agent-Based-Simulation-Model/bin/sim"
LOG_PATH = "/media/Big-Boy/Nextcloud/Personal-Programming/python/Modeling-Tools-Data/models/LTE/output_log.txt"
# LOG_PATH = "/home/alex/Nextcloud/Personal-Programming/python/Modeling-Tools-Data/models/LTE/output_log.txt"

def get_trajectories(params: Params):
    overseer = Overseer(EXE_PATH, params, LOG_PATH)
    sim_finished = False
    while not sim_finished:
        sim_finished = overseer.step()
        traj, t = overseer.get_data()
        yield traj, t
