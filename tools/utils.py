import os

import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # args.rank = int(os.environ['SLURM_PROCID'])
        # args.gpu = args.rank % torch.cuda.device_count()
        if torch.cuda.device_count() == 1:
            print('Not using distributed mode')
            args.distributed = False
            return
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    if args.distributed:
        local_rank = int(os.environ['SLURM_LOCALID'])
        args.local_rank = local_rank
        port = str(args.port)  # 自己指定0-65535之间
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        if '[' in node_list:
            beg = node_list.find('[')
            pos1 = node_list.find('-', beg)
            if pos1 < 0:
                pos1 = 1000
            pos2 = node_list.find(',', beg)
            if pos2 < 0:
                pos2 = 1000
            node_list = node_list[:min(pos1, pos2)].replace('[', '')
        addr = node_list[8:].replace('-', '.')
        os.environ['MASTER_PORT'] = port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        #args.device = torch.device("cuda", args.local_rank)
        args.dist_backend = 'nccl'
        host_addr_full = 'tcp://' + addr + ':' + port
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=host_addr_full,
                                             world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)