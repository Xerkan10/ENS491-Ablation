import torch
from parameters import args_parser
import argparse
from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex
from torch.multiprocessing import set_start_method
import multiprocessing as mp
import itertools
import torch.multiprocessing as mpcuda
import numpy as np
import time
from run import main_thread

"""
This main is for device and thread management. Any parameter in given args as list,
will then convert into list of paramaters to perform a grid search of parameters. 

These combinations will run in paralel threads with a maximum allowed instances for single gpu.
"""

if __name__ == '__main__':
    args = args_parser()
    worker_per_device = args.worker_per_device
    cuda = args.cuda
    cuda_info = None
    
    # Check for available devices
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'
        cuda = False
        print('Using MPS (Metal Performance Shaders) device')
    else:
        device_type = 'cpu'
        cuda = False
        print('No GPU found, using CPU device')
    
    Process = mpcuda.Process if cuda else mp.Process
    available_gpus = torch.cuda.device_count() - len(args.excluded_gpus) if cuda else 0
    max_active_user = available_gpus * worker_per_device if cuda else worker_per_device
    first_gpu_share = np.repeat(worker_per_device, torch.cuda.device_count())
    first_gpu_share[args.excluded_gpus] = 0
    combinations = []
    work_load = []
    simulations = []
    w_parser = argparse.ArgumentParser()
    started = 0
    excluded_args = ['excluded_gpus','lr_decay']
    for arg in vars(args):
        arg_type = type(getattr(args, arg))
        if arg_type == list and arg not in excluded_args:
            work_ = [n for n in getattr(args, arg)]
            work_load.append(work_)
    for t in itertools.product(*work_load):
        combinations.append(t)
    print('Number of simulations is :',len(combinations))
    for combination in combinations:
        w_parser = argparse.ArgumentParser()
        listC = 0
        for arg in vars(args):
            arg_type = type(getattr(args, arg))
            if arg_type == list and arg not in excluded_args:
                new_type = type(combination[listC])
                w_parser.add_argument('--{}'.format(arg), type=new_type, default=combination[listC], help='')
                listC += 1
            else:
                val = getattr(args, arg)
                new_type = type(getattr(args, arg))
                w_parser.add_argument('--{}'.format(arg), type=new_type, default=val, help='')

        if cuda:
            if started < max_active_user:
                selected_gpu = np.argmax(first_gpu_share)
                first_gpu_share[selected_gpu] -= 1
            else:
                nvmlInit()
                cuda_info = [nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i))
                             for i in range(torch.cuda.device_count())]
                cuda_memory = np.zeros(torch.cuda.device_count())
                for i, gpu in enumerate(cuda_info):
                    if i not in args.excluded_gpus:
                        cuda_memory[i] = gpu.free
                selected_gpu = np.argmax(cuda_memory)
            print('Process {} assigned with gpu:{}'.format(started, selected_gpu))
            w_parser.add_argument('--gpu_id', type=int, default=selected_gpu,
                                  help='cuda device selected')  # assign gpu for the work
        else:
            w_parser.add_argument('--gpu_id', type=int, default=-1, help='cpu selected')  # assign gpu for the work

        w_args = w_parser.parse_args()
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        process = Process(target=main_thread, args=(w_args,))
        process.start()
        simulations.append(process)
        started += 1

        while not len(simulations) < max_active_user:
            for i, process_data in enumerate(simulations):
                if not process_data.is_alive():
                    # remove from processes
                    p = simulations.pop(i)
                    del p
                    time.sleep(10)