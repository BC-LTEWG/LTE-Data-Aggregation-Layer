import logging
logger = logging.getLogger(__name__)
import os
from .parameters import Params
from .Overseer import Overseer
from typing import Tuple
import numpy as np
import subprocess
import json
from scipy.linalg import inv

print("hello?")

# REPLACE WITH THE FILE PATH TO YOUR OWN BINARY. IF USING WINDOWS, MAKE SURE YOU USE DOUBLE FORWARD SLASHES (e.g. C:\\Users\\...)
# devin
EXE_PATH = "/home/alex/github/Labor-Time-Economy-Simulation/bin/sim"
# me
# EXE_PATH = "/home/alex/github/temp/Labor-Time-Economy-Simulation/bin/sim"
# LOG_PATH = "/media/Big-Boy/Nextcloud/Personal-Programming/python/Modeling-Tools-Data/models/LTE-Data-Aggregation-Layer/output_log.txt"
LOG_PATH = "/home/alex/Nextcloud/Personal-Programming/python/Modeling-Tools-Data/models/LTE-Data-Aggregation-Layer/output_log.txt"

def get_trajectories(params: Params, event_queue):
    overseer = Overseer(EXE_PATH, params, LOG_PATH)
    try:
        sim_finished = False
        while not sim_finished:
            sim_finished = overseer.step()
            traj = overseer.get_data()
            yield traj
    finally:
        overseer.collector.stop()
