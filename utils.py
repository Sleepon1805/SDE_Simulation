import torch

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
