import sys
import torch
import matplotlib.pyplot as plt
from time import time
from typing import List, Dict, Tuple
from sklearn.linear_model import LinearRegression

from sde_lib import SDE, GeometricBrownianMotion, OrnsteinUhlenbeck, CoxIngersollRoss
from approximation_methods import get_approximation_method

DEFAULT_BATCH_SIZE = 5000  # 20000 (num_simulations) x 32768 (timesteps) still pass on my Radeon RX 6700 XT


class ConvergenceRateCalculator:
    def __init__(self, sde: SDE, approximation_methods: List[str], dt_grid: List[float], p_values: List[int],
                 num_simulations: int):
        self.sde = sde
        self.approximation_methods = approximation_methods
        self.dt_grid = dt_grid
        self.p_values = p_values
        self.num_simulations = num_simulations

        self.num_steps_grid = [int(sde.T / dt) for dt in dt_grid]

        # for gpu calculations
        self.batch_size = min(DEFAULT_BATCH_SIZE, num_simulations)

        # for plotting
        self.colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        self.linestyles = ['-', '--', ':']

    def test_different_sde_params(self, param_name: str, param_values: List[float], use_exact_solution: True, visualize=True):
        convergence_rates = {}
        for param_value in param_values:
            print(f"Testing {param_name} = {param_value}")
            setattr(self.sde, param_name, param_value)
            convergence_rates[param_value] = self.calculate_convergence_rates(use_exact_solution, visualize)

        if visualize:
            for i, method_name in enumerate(self.approximation_methods):
                for j, p in enumerate(self.p_values):
                    plt.plot(
                        param_values,
                        [convergence_rates[param_value][(method_name, p)] for param_value in param_values],
                        label=f"{method_name} - p = {p}", color=self.colors[i], ls=self.linestyles[j]
                    )
            plt.title(f'{"Pseudo " if not use_exact_solution else ""}Approximation Errors for\n {self.sde}')
            plt.xlabel(param_name)
            plt.ylabel('Convergence Rate')
            plt.legend()
            plt.show()
        return convergence_rates

    def calculate_convergence_rates(self, use_exact_solution: True, visualize=True):
        adapted_sde_name = str(self.sde).replace(
            f"N={self.sde.N}",
            f"N=[{max(self.num_steps_grid)} ... {min(self.num_steps_grid)}]"
        )
        print(f'SDE: {adapted_sde_name}')

        errors = []
        assert self.num_simulations % self.batch_size == 0, "float number of batches not implemented yet"
        num_batches = self.num_simulations // self.batch_size
        ttime = time()
        for batch_num in range(num_batches):
            print(f'Batch {batch_num + 1}/{num_batches}')
            errors.append(self._calculate_errors(self.batch_size, use_exact_solution))
        errors = {key: sum([error[key] for error in errors]) / num_batches for key in errors[0].keys()}
        print(f"Total simulation time: {time() - ttime:.2f}s")

        if visualize:
            # plot lp errors
            for i, method_name in enumerate(self.approximation_methods):
                for j, p in enumerate(self.p_values):
                    plt.loglog(self.dt_grid, [errors[(method_name, p, N)] for N in self.num_steps_grid],
                               label=f"{method_name} - p = {p}", color=self.colors[i], ls=self.linestyles[j])
            plt.title(f'{"Pseudo " if not use_exact_solution else ""}Approximation Errors for\n {self.sde}')
            plt.xlabel('$\Delta t$')
            plt.ylabel('Error (e($\Delta t$))')
            plt.legend()
            plt.show()

        convergence_rates = self._calculate_convergence_rates_from_errors(errors)

        return convergence_rates

    def _calculate_errors(self, num_simulations: int, use_exact_solution: bool) -> Dict[Tuple[str, int, int], float]:
        errors = {
            (method_name, p, N): 0
            for method_name in self.approximation_methods for p in self.p_values for N in self.num_steps_grid
        }

        # sample brownian motion with the highest discretization
        highest_res_N = max(self.num_steps_grid) * (2 ** 2)
        highest_res_sde = self.sde.copy()
        highest_res_sde.update_time_discretization(num_steps=highest_res_N)
        brownian_motion = highest_res_sde.sample_brownian_motion(num_simulations=num_simulations)

        # calculate exact solutions as approximations with higher discretization
        if use_exact_solution:
            print(f"Simulating exact solutions with N = {highest_res_N}", end="")
            ttime = time()
            all_exact_solutions = highest_res_sde.exact_solutions(brownian_motion)
            all_exact_solutions = {
                method_name: all_exact_solutions for method_name in self.approximation_methods
            }
            print(f" ({time() - ttime:.2f}s)")
        else:
            print(f"Simulating pseudo exact solutions for N = {highest_res_N}", end="")
            ttime = time()
            all_exact_solutions = {}
            for method_name in self.approximation_methods:
                all_exact_solutions[method_name] = get_approximation_method(method_name)(
                    sde=highest_res_sde, W=brownian_motion)
            print(f" ({time() - ttime:.2f}s)")

        print(f"Running simulations for N in {self.num_steps_grid}:")
        for N in self.num_steps_grid:
            ttime = time()
            self.sde.update_time_discretization(num_steps=N)
            for method_name in self.approximation_methods:
                exact_solutions = all_exact_solutions[method_name][::int(highest_res_N/N)]
                reduced_brownian_motion = brownian_motion[::int(highest_res_N/N)]
                approximations = get_approximation_method(method_name)(self.sde, reduced_brownian_motion)
                for p in self.p_values:
                    errors[(method_name, p, N)] += self._calculate_lp_error(
                        exact_solutions, approximations, p=p)
            print(f"{time() - ttime:.2f}s + ", end="")
        print("")
        return errors

    @staticmethod
    def _calculate_lp_error(solutions, calc_approximations, p):
        """
        Calculate L^p error between sampled solutions and approximations
        :param solutions: torch.Tensor: exact solutions of shape (num_timesteps[, num_simulations])
        :param calc_approximations: torch.Tensor: approximations of shape (num_timesteps[, num_simulations])
        :param p: float: p in L^p norm
        :return: float: strong L^p error
        """
        assert p >= 1, "p \in [1, \infty)"
        error = torch.max(torch.mean(torch.abs(solutions - calc_approximations) ** p, dim=1) ** (1/p))
        return error.cpu().item()

    def _calculate_convergence_rates_from_errors(self, error_values):
        """
        Calculate convergence rate from given errors and time discretizations.
        """
        print("---")
        convergence_rates = {}
        for method_name in self.approximation_methods:
            for p in self.p_values:
                x = torch.log(torch.tensor(self.dt_grid)).reshape(-1, 1)
                y = torch.log(torch.tensor([error_values[(method_name, p, N)] for N in self.num_steps_grid]))
                reg = LinearRegression().fit(x, y)
                convergence_rates[(method_name, p)] = reg.coef_.item()
                print(f"{method_name} convergence rate for p = {p}: {convergence_rates[(method_name, p)]}")
            print("---")
        return convergence_rates


if __name__ == '__main__':
    APPROXIMATION_METHODS = [
        # 'Euler-Maruyama',
        # 'Milstein',
        # 'Truncated Milstein',
        'Alfonsi Implicit (3)',
        'Alfonsi Implicit (4)',
        'Alfonsi Explicit 0',  # lambda=0 correspond to (4)
        'Alfonsi Explicit sigma^2/4',  # lambda = sigma ^ 2 / 4 correspond to (3)
        # 'Runge-Kutta',
        # 'Time Adaptive 0',
    ]

    # SDE = GeometricBrownianMotion(time_horizon=1, num_steps=1, x0=1, mu=2, sigma=1)
    SDE = CoxIngersollRoss(time_horizon=1, num_steps=1, x0=1, a=1, b=1, sigma=2)
    # SDE = OrnsteinUhlenbeck(time_horizon=1, num_steps=1, x0=1, mu=0, theta=1, sigma=1

    calc = ConvergenceRateCalculator(
        sde=SDE,
        approximation_methods=APPROXIMATION_METHODS,
        dt_grid=[2 ** i for i in range(-14, -4)],
        p_values=[1],
        num_simulations=20000,
    )
    # calc.calculate_convergence_rates(use_exact_solution=False, visualize=True)
    calc.test_different_sde_params('sigma', [0.5, 1, 1.5, 2, 2.5, 3], use_exact_solution=False, visualize=True)
