import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

from utils import on_device, SEED, neg_part


class SDE(ABC):
    def __init__(self, time_horizon, num_steps, x0):
        # time discretization
        self.T = time_horizon
        self.N = num_steps
        self.dt = self.T / self.N
        self.ts = torch.linspace(0, self.T, self.N)[:, None]

        # initial value
        self.x0 = x0

    @abstractmethod
    def drift(self, X_t, t):
        """
        Abstract method for drift coefficient
        :param X_t: float: current value of the process
        :param t: float: current time
        :return: float: drift coefficient
        """
        pass

    @abstractmethod
    def diffusion(self, X_t, t):
        """
        Abstract method for diffusion coefficient
        :param X_t: float: current value of the process
        :param t: float: current time
        :return: float: diffusion coefficient
        """
        pass

    @abstractmethod
    @on_device
    def exact_solutions(self, W: torch.Tensor) -> torch.Tensor:
        """
        Sample exact solution paths of the SDE
        :param W: torch.Tensor: brownian motion paths of shape (num_timesteps, num_simulations) to simulate SDE
        :return: torch.Tensor: Paths of shape (num_timesteps, num_simulations)
        """
        pass

    def sample_brownian_motion(self, num_simulations=1, seed=None):
        """
        Sample random Brownian Motion with given time discretization
        :param num_simulations: int: number of sampled BBs
        :param seed: int: seed to get control over sampled BB
        :return: torch.Tensor: BB paths of shape (num_timesteps, num_simulations)
        """
        if seed is not None:
            torch.manual_seed(seed)
        dW = self.dt ** 0.5 * torch.randn(self.N, num_simulations)
        dW[0] = 0
        W = torch.cumsum(dW, dim=0)
        return W

    def update_time_discretization(self, num_steps=None, dt=None, ts=None):
        """
        Update time discretization of SDE instance. Only one of num_steps, dt or ts must be specified.
        :param num_steps: int: number of time steps
        :param dt: float: time step
        :param ts: torch.Tensor: time grid
        """
        if num_steps is not None:
            self.N = num_steps
            self.dt = self.T / self.N
            self.ts = torch.linspace(0, self.T, self.N)[:, None]
        elif dt is not None:
            self.dt = dt
            self.N = int(self.T / self.dt)
            self.ts = torch.linspace(0, self.T, self.N)[:, None]
        elif ts is not None:
            self.ts = ts.reshape(-1, 1)
            self.N = len(ts)
            self.dt = int(ts[1] - ts[0])
        else:
            raise ValueError("Either num_steps, dt or ts must be specified")

    def to_str(self, hide: str | List[str] = None):
        sde_name = str(self)
        if hide is None:
            return sde_name

        if isinstance(hide, str):
            hide = [hide]

        for param_name in hide:
            sde_name = sde_name.replace(f"{param_name}={getattr(self, param_name)}, ", "")
        return sde_name

    def __repr__(self):
        """
        String representation of SDE instance
        """
        return f"{type(self).__name__}(time_horizon={self.T}, num_steps={self.N}, x0={self.x0})"

    def copy(self):
        """
        Create a copy of the SDE instance
        """
        return deepcopy(self)

    def to(self, device):
        """
        Move SDE instance to device
        """
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                self.__setattr__(attr_name, attr_value.to(device))
        return self


class GeometricBrownianMotion(SDE):
    def __init__(self, time_horizon, num_steps, x0, mu, sigma):
        super().__init__(time_horizon, num_steps, x0)
        self.mu = mu
        self.sigma = sigma

    def drift(self, X_t, t):
        return self.mu * X_t

    def diffusion(self, X_t, t):
        return self.sigma * X_t

    @on_device
    def exact_solutions(self, W):
        X = self.x0 * torch.exp((self.mu - 0.5 * self.sigma ** 2) * self.ts + self.sigma * W)
        return X

    def __repr__(self):
        return f"Geometric Brownian Motion with T={self.T}, N={self.N}, x0={self.x0}, mu={self.mu}, sigma={self.sigma}"


class OrnsteinUhlenbeck(SDE):
    def __init__(self, time_horizon, num_steps, x0, mu, theta, sigma):
        super().__init__(time_horizon, num_steps, x0)
        assert sigma > 0, "Assumptions on SDE parameters are not satisfied"
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    def drift(self, X_t, t):
        return self.theta * (self.mu - X_t)

    def diffusion(self, X_t, t):
        return self.sigma

    @on_device
    def exact_solutions(self, W):
        dW = torch.diff(W, dim=0)

        int_terms = torch.zeros_like(W, device=W.device)
        for i in range(1, self.N):
            int_terms[i] = torch.sum(torch.exp(-self.theta * (self.ts[i] - self.ts[:i])) * dW[:i], dim=0)

        X = (self.x0 * torch.exp(-self.theta * self.ts) + self.mu * (1 - torch.exp(-self.theta * self.ts))
             + self.sigma * int_terms)
        return X

    def __repr__(self):
        return (f"Ornstein-Uhlenbeck with "
                f"T={self.T}, N={self.N}, x0={self.x0}, mu={self.mu}, theta={self.theta}, sigma={self.sigma}")


class CoxIngersollRoss(SDE):
    def __init__(self, time_horizon, num_steps, x0, a, b, sigma):
        super().__init__(time_horizon, num_steps, x0)
        assert a >= 0 and b >= 0 and sigma > 0, "Assumptions on SDE parameters are not satisfied"
        self.a = a
        self.b = b
        self.sigma = sigma
        self.delta = 4 * self.a / self.sigma ** 2

    def drift(self, X_t, t):
        return self.a - self.b * X_t

    def diffusion(self, X_t, t):
        return self.sigma * (X_t ** 0.5)

    @on_device
    def exact_solutions(self, W):
        if self.a == 1 and self.sigma == 2:
            """
            dX_t = (1 - bX_t) dt + 2 sqrt(X_t) dW_t
            https://arxiv.org/pdf/1601.01455.pdf (Proposition 1)
            """
            U = OrnsteinUhlenbeck(
                self.T, self.N, self.x0 ** 0.5, mu=0, theta=self.b/2, sigma=1
            ).exact_solutions(W).to(W.device)
            inf_inputs = torch.exp(self.b * self.ts / 2) * U
            X = torch.zeros_like(W)
            X[0] = self.x0
            for i in range(self.N-1):
                X[i+1] = (U[i+1] + torch.exp(-self.b * self.ts[i] / 2) *
                          (neg_part(torch.min(inf_inputs[:i+1], dim=0)[0]))) ** 2
        else:
            raise NotImplementedError("Exact solution is not implemented or not known")
        return X

    def __repr__(self):
        return (f"Cox-Ingersoll-Ross with "
                f"T={self.T}, N={self.N}, x0={self.x0}, a={self.a}, b={self.b}, sigma={self.sigma}")


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
        brownian_motion = sde.sample_brownian_motion(num_simulations=1, seed=SEED)

        for NUM_STEPS in num_steps_grid:
            sde.update_time_discretization(num_steps=NUM_STEPS)
            reduced_brownian_motion = brownian_motion[::int(max(num_steps_grid)/NUM_STEPS)]
            exact_solution = sde.exact_solutions(reduced_brownian_motion)
            plt.plot(sde.ts, exact_solution, label=f'N={NUM_STEPS}', linewidth=1)

        plt.title(f'Sampled exact solution paths for\n {sde}')
        plt.ylabel('Y(t)')
        plt.xlabel('t')
        plt.legend()
        plt.show()
