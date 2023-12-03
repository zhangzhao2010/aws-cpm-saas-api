import torch
from typing_extensions import TypedDict
class ConfigMap(TypedDict):
    rank : int
    local_rank : int
    world_size : int
    local_size : int
    zero_level : int
    pipe_size : int
    num_micro_batches : int
    calc_stream : torch.cuda.Stream
    load_stream : torch.cuda.Stream
    load_event : torch.cuda.Event
    barrier_stream : torch.cuda.Stream
    loss_scale_factor : float
    loss_scale_steps : int
    topology : 'topology'
    gradient_inspect : bool
    initialized : bool

    comm : 'NCCLCommunicator'

config = ConfigMap(rank=0, local_rank=0, world_size=1, initialized=False)

def rank():
    """
    Returns the global rank of the current process. (0 ~ world_size-1)
    """
    return config['rank']

def world_size():
    """
    Returns the total number of workers across all nodes.
    """
    return config['world_size']
