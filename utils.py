import torch
from tqdm import tqdm

SEED = 137
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')


def on_device(func):
    def wrapper(*args, **kwargs):
        args = [arg.to(DEVICE) for arg in args if hasattr(arg, 'to')]
        kwargs = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in kwargs.items()}
        result = func(*args, **kwargs)
        args = [arg.to('cpu') for arg in args if hasattr(arg, 'to')]
        kwargs = {k: v.to('cpu') if hasattr(v, 'to') else v for k, v in kwargs.items()}
        return result.to('cpu')
    return wrapper


def run_batched(func, arg_name, batch_size, batch_dim, *args, **kwargs):
    assert arg_name in kwargs, f'Argument {arg_name} not found in kwargs'
    assert isinstance(kwargs[arg_name], torch.Tensor), f'Argument {arg_name} is not a torch.Tensor'
    arg = kwargs[arg_name]

    if arg.shape[batch_dim] <= batch_size:
        return func(*args, **kwargs)

    arg_batched = torch.split(arg, batch_size, dim=batch_dim)
    results = []
    for arg_batch in tqdm(arg_batched):
        kwargs[arg_name] = arg_batch
        results.append(func(*args, **kwargs))
    results = torch.cat(results, dim=batch_dim)
    return results


def pos_part(x: torch.Tensor):
    return torch.maximum(x, torch.tensor(0, device=x.device))
