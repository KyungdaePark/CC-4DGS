#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def find_best_iteration_from_log(log_file_path):
    """
    log_file_path : e.g. scene.model_path + '/log_metrics.txt'
    Returns best_iter, best_psnr (int, float)
    """
    best_iter = None
    best_psnr = -1

    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Log file not found: {log_file_path}")
        return None, None

    # lines[0] might be header: "iteration, mPSNR, nPoints\n"
    # lines[1:] -> data

    # Optionally, check if the first line is a header
    # If you always have a header, skip lines[0]
    start_idx = 0
    if 'iteration' in lines[0]:
        start_idx = 1

    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        # e.g. "300, 27.4936, 59725"
        parts = [p.strip() for p in line.split(',')]
        # if len(parts) < 2:
        #     continue
        iteration_str, mpsnr_str = parts[0], parts[1]
        # optional: nPoints_str = parts[2] if you need it

        it = int(iteration_str)
        ps = float(mpsnr_str)

        if ps > best_psnr:
            best_psnr = ps
            best_iter = it

    return best_iter, best_psnr