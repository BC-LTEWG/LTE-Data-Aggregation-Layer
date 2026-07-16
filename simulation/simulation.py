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

def get_trajectories(params: Params, event_queue):
    overseer = Overseer(params.exe_path, params)
    try:
        sim_finished = False
        while not sim_finished:
            sim_finished = overseer.step()
            traj = overseer.get_data()
            yield traj
    finally:
        overseer.collector.stop()
