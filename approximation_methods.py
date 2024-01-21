import torch
from sympy import Symbol, diff, lambdify
from matplotlib import pyplot as plt
from functools import partial
from tqdm import tqdm
import re

from sde_lib import SDE, GeometricBrownianMotion, OrnsteinUhlenbeck, CoxIngersollRoss
from utils import on_device, SEED, pos_part

APPROXIMATION_METHODS = [
    'Euler-Maruyama',
    'Milstein',
    'Truncated Milstein',
    'Alfonsi (3)',
    'Alfonsi (4)',
    'Alfonsi E(0)',  # lambda=0 correspond to (4)
    'Alfonsi E(sigma^2/4)',  # lambda = sigma ^ 2 / 4 correspond to (3)
    'Runge-Kutta',
]


def get_approximation_method(name: str):
    """
    Get approximation method by name
    :param name: str: name of approximation method
    :return: function: approximation method
    """
    if name == 'Euler-Maruyama':
        return euler_maruyama
    elif name == 'Milstein':
        return milstein
    elif name == 'Truncated Milstein':
        return truncated_milstein
    elif name == 'Alfonsi (3)':
        return alfonsi_implicit_euler_3
    elif name == 'Alfonsi (4)':
        return alfonsi_implicit_euler_4
    elif 'Alfonsi E(' in name:
        lambda_param = name.split('Alfonsi E(')[1]
        lambda_param = lambda_param.split(')')[0]
        return partial(alfonsi_explicit_euler, lambda_param=lambda_param)
    elif name == 'Runge-Kutta':
        return runge_kutta
    elif 'Time Adaptive ' in name:
        lamda_param = float(name.split('Time Adaptive ')[1])
        return partial(time_adaptive, lambda_param=lamda_param)
    else:
        raise NotImplementedError(f'Approximation method {name} not implemented')


@on_device
def euler_maruyama(sde: SDE, W: torch.Tensor):
    dW = torch.diff(W, dim=0)
    X = torch.zeros_like(W, device=W.device)
    X[0] = sde.x0
    for i in range(sde.N - 1):
        X[i + 1] = X[i] + sde.drift(X[i], sde.ts[i]) * sde.dt + sde.diffusion(X[i], sde.ts[i]) * dW[i]
    return X


@on_device
def milstein(sde: SDE, W: torch.Tensor):
    assert all(sde.diffusion(sde.x0, 0) == sde.diffusion(sde.x0, t) for t in sde.ts), \
        "Milstein method requires diffusion to be independent of time"

    dW = torch.diff(W, dim=0)
    X = torch.zeros_like(W, device=W.device)
    X[0] = sde.x0

    if isinstance(sde, CoxIngersollRoss):
        for i in range(sde.N - 1):
            X[i + 1] = (X[i] + sde.drift(X[i], sde.ts[i]) * sde.dt + sde.diffusion(X[i], sde.ts[i]) * dW[i]
                        + 0.25 * sde.sigma ** 2 * (dW[i] ** 2 - sde.dt))
    else:
        x = Symbol('X')
        diffusion_derivative = lambdify(x, diff(sde.diffusion(x, t=0), x), 'numpy')
        for i in range(sde.N - 1):
            X[i + 1] = (X[i] + sde.drift(X[i], sde.ts[i]) * sde.dt + sde.diffusion(X[i], sde.ts[i]) * dW[i]
                        + 0.5 * sde.diffusion(X[i], sde.ts[i]) * diffusion_derivative(X[i]) * (dW[i] ** 2 - sde.dt))
    return X


@on_device
def truncated_milstein(sde: SDE, W: torch.Tensor):
    if isinstance(sde, CoxIngersollRoss):
        """
        https://arxiv.org/pdf/1608.00410.pdf section 5.1 (31)
        """
        assert sde.sigma == 2, "Truncated Milstein method only implemented for sigma = 2"
        dW = torch.diff(W, dim=0)
        X = torch.zeros_like(W, device=W.device)
        X[0] = sde.x0

        dt = torch.tensor(sde.dt, device=W.device)
        for i in range(sde.N - 1):
            nonlinear_part = torch.maximum(dt ** 0.5, torch.maximum(dt, X[i]) ** 0.5 + dW[i]) ** 2
            X[i + 1] = pos_part(nonlinear_part + (sde.delta - 1 - sde.b * X[i]) * dt)
    else:
        raise NotImplementedError(f'Truncated Milstein method not implemented for {sde}')
    return X


@on_device
def alfonsi_implicit_euler_3(sde: SDE, W: torch.Tensor):
    """
    https://cermics.enpc.fr/~alfonsi/SC_preprint.pdf Implicit (3) with notes under (30)
    """
    assert isinstance(sde, CoxIngersollRoss), \
        "Alfonsi implicit Euler method (3) only works for CIR process"
    dW = torch.diff(W, dim=0)
    X = torch.zeros_like(W, device=W.device)
    X[0] = sde.x0
    for i in range(sde.N - 1):
        discriminant = (sde.sigma ** 2 * dW[i] ** 2 +
                        4 * (X[i] + (sde.a - sde.sigma ** 2 / 2) * sde.dt) * (1 + sde.b * sde.dt))
        denominator = torch.where(discriminant >= 0, sde.sigma * dW[i] + discriminant ** 0.5, 0)
        X[i + 1] = (denominator / (2 + 2 * sde.b * sde.dt)) ** 2
    return X


@on_device
def alfonsi_implicit_euler_4(sde: SDE, W: torch.Tensor):
    """
    https://cermics.enpc.fr/~alfonsi/SC_preprint.pdf (4) with notes under (30)
    """
    assert isinstance(sde, CoxIngersollRoss), \
        "Alfonsi implicit Euler method (4) only works for CIR process"
    dW = torch.diff(W, dim=0)
    X = torch.zeros_like(W, device=W.device)
    X[0] = sde.x0
    for i in range(sde.N - 1):
        discriminant = ((0.5 * sde.sigma * dW[i] + X[i] ** 0.5) ** 2 +
                        (2 + sde.b * sde.dt) * (sde.a - sde.sigma ** 2 / 4) * sde.dt)
        denominator = torch.where(discriminant >= 0, 0.5 * sde.sigma * dW[i] + X[i] ** 0.5 + discriminant ** 0.5, 0)
        X[i + 1] = (denominator / (2 + sde.b * sde.dt)) ** 2
    return X


@on_device
def alfonsi_explicit_euler(sde: SDE, W: torch.Tensor, lambda_param: float):
    """
    https://cermics.enpc.fr/~alfonsi/SC_preprint.pdf (5) with notes under (30)
    """
    assert isinstance(sde, CoxIngersollRoss), \
        "Alfonsi explicit scheme E(lambda) only works for CIR process"

    if isinstance(lambda_param, float):
        lambda_value = lambda_param
    elif isinstance(lambda_param, str):
        if 'sigma^2/' in lambda_param:
            lambda_value = sde.sigma / float(lambda_param.split('sigma^2/')[-1])
        else:
            lambda_value = float(lambda_param)
    else:
        raise ValueError(f'lambda_param must be float or str, but is {type(lambda_param)}')

    dW = torch.diff(W, dim=0)
    X = torch.zeros_like(W, device=W.device)
    X[0] = sde.x0
    factor = 1 - 0.5 * sde.b * sde.dt
    # factor = (1 - sde.b * sde.dt) ** 0.5
    for i in range(sde.N - 1):
        X[i + 1] = (
            (1 - 0.5 * sde.b * sde.dt) * X[i] ** 0.5 + sde.sigma * dW[i] / (2 * factor)
        ) ** 2 + (
            (sde.a - sde.sigma ** 2 / 4) * sde.dt + lambda_value * (dW[i] ** 2 - sde.dt)
        )
        X[i + 1] = torch.max(X[i+1], torch.tensor(0, device=W.device))
    return X


@on_device
def runge_kutta(sde: SDE, W: torch.Tensor):
    assert all(sde.drift(sde.x0, 0) == sde.drift(sde.x0, t) for t in sde.ts), \
        "Runge-Kutta method requires drift to be independent of time"
    assert all(sde.diffusion(sde.x0, 0) == sde.diffusion(sde.x0, t) for t in sde.ts), \
        "Runge-Kutta method requires diffusion to be independent of time"

    dW = torch.diff(W, dim=0)
    X = torch.zeros_like(W, device=W.device)
    X[0] = sde.x0
    for i in range(0, sde.N - 1):
        a = sde.drift(X[i], 0)
        b = sde.diffusion(X[i], 0)
        korr_X = X[i] + a * sde.dt + b * sde.dt ** 0.5
        X[i + 1] = (X[i] + a * sde.dt + b * dW[i] +
                    0.5 * (sde.diffusion(korr_X, 0) - b) * (dW[i] ** 2 - sde.dt) * (sde.dt ** -0.5))
    return X


@on_device
def time_adaptive(sde: SDE, W: torch.Tensor, lambda_param: float):
    assert lambda_param >= 1

    if isinstance(sde, CoxIngersollRoss):
        """
        https://arxiv.org/pdf/1601.01455.pdf section 4 (algorithm (26))
        """
        assert W.shape[1] == 1, "Time adaptive method only implemented for single simulation"

        chosen_timesteps = sde.ts[[0, -1]].squeeze().type(torch.float64)
        W_values = W[[0, -1]].squeeze()
        for n in tqdm(range(2, sde.N)):
            min_value = torch.min(W_values)
            time_diffs = torch.diff(chosen_timesteps)
            h = torch.min(time_diffs)
            assert h > 0  # wtf
            epsilon = (-lambda_param * h * torch.log(h))

            probs = torch.zeros(n-1)
            for k in range(0, n-1):  # notation according to paper
                T_k = time_diffs[k]
                probs[k] = T_k / ((W_values[k] - min_value + epsilon) * (W_values[k+1] - min_value + epsilon))
            k = torch.argmax(probs)
            chosen_ts = (chosen_timesteps[k+1] + chosen_timesteps[k]).reshape(1) / 2

            nearest_index = torch.floor(torch.tensor(chosen_ts / sde.dt)).int()
            W_value = W[nearest_index].squeeze() + (chosen_ts % sde.dt) ** 0.5 * torch.randn(1, device=W.device)

            chosen_timesteps, sorted_indices = torch.sort(torch.cat([chosen_timesteps, chosen_ts]))
            W_values = torch.cat([W_values, W_value])[sorted_indices]

        X = torch.zeros_like(W, device=W.device)
        raise NotImplementedError   # TODO
    else:
        raise NotImplementedError(f'Time adaptive method not implemented for {sde}')
    return X


if __name__ == '__main__':
    plt.figure(figsize=(15, 9))
    T = 1  # time horizon
    X0 = 1  # initial value
    num_steps_grid = [2 ** i for i in range(5, 11)]

    sdes = [
        # GeometricBrownianMotion(time_horizon=T, num_steps=max(num_steps_grid), x0=X0, mu=2, sigma=1),
        # OrnsteinUhlenbeck(time_horizon=T, num_steps=max(num_steps_grid), x0=X0, mu=0, theta=1, sigma=1),
        CoxIngersollRoss(time_horizon=T, num_steps=max(num_steps_grid), x0=X0, a=1, b=1, sigma=2),
    ]

    for single_sde in sdes:
        # Brownian motion
        brownian_motion = single_sde.sample_brownian_motion(num_simulations=1, seed=SEED)
        # plt.plot(single_sde.ts, brownian_motion, label="BB", linewidth=0.3)

        # Exact solution
        try:
            exact_solution = single_sde.exact_solutions(brownian_motion)
            plt.plot(single_sde.ts, exact_solution, label="Exact ($Y_t$)", linewidth=0.3)
        except AssertionError as e:
            print(e)
        except NotImplementedError as e:
            print(e)

        # Approximations
        for method_name in APPROXIMATION_METHODS:
            try:
                approximation = get_approximation_method(method_name)(single_sde, brownian_motion)
                plt.plot(single_sde.ts, approximation, label=f"{method_name} approximation ($Y_t$)", linewidth=0.3)
            except AssertionError as e:
                print(e)
            except NotImplementedError as e:
                print(e)
        plt.title(f'Approximations of\n {single_sde}')
        plt.xlabel('t')
        plt.legend()
        plt.show()
