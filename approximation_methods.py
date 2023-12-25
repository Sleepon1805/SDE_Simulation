import torch
from sympy import Symbol, diff, lambdify
from matplotlib import pyplot as plt

from sde_lib import SDE, GeometricBrownianMotion, OrnsteinUhlenbeck, CoxIngersollRoss
from utils import on_device, SEED

APPROXIMATION_METHODS = ['Euler-Maruyama', 'Milstein', 'Runge-Kutta']  # list of methods to use


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
    elif name == 'Runge-Kutta':
        return runge_kutta
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


if __name__ == '__main__':
    T = 1  # time horizon
    X0 = 1  # initial value
    num_steps_grid = [2 ** i for i in range(5, 11)]

    sdes = [
        GeometricBrownianMotion(time_horizon=T, num_steps=max(num_steps_grid), x0=X0, mu=2, sigma=1),
        OrnsteinUhlenbeck(time_horizon=T, num_steps=max(num_steps_grid), x0=X0, mu=0, theta=1, sigma=1),
        CoxIngersollRoss(time_horizon=T, num_steps=max(num_steps_grid), x0=X0, a=1, b=0, sigma=2),
    ]

    for sde in sdes:
        # Brownian motion
        brownian_motion = sde.sample_brownian_motion(num_simulations=1, seed=SEED)

        # Exact solution
        exact_solution = sde.exact_solutions(brownian_motion)
        plt.plot(sde.ts, brownian_motion, label="BB", linewidth=0.3)
        plt.plot(sde.ts, exact_solution, label="Exact ($Y_t$)", linewidth=0.3)

        # Approximations
        for method_name in APPROXIMATION_METHODS:
            approximation = get_approximation_method(method_name)(sde, brownian_motion)
            plt.plot(sde.ts, approximation, label=f"{method_name} approximation ($Y_t$)", linewidth=0.3)
        plt.title(f'Approximations of {sde}')
        plt.xlabel('t')
        plt.legend()
        plt.show()
