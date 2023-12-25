import torch
import matplotlib.pyplot as plt
from time import time
from typing import List, Dict, Tuple
from sklearn.linear_model import LinearRegression

from sde_lib import SDE, GeometricBrownianMotion, OrnsteinUhlenbeck, CoxIngersollRoss
from approximation_methods import get_approximation_method, APPROXIMATION_METHODS
from utils import on_device, SEED


class ConvergenceRateCalculator:
    def __init__(self, sde: SDE, approximation_methods: List[str], dt_grid: List[float], p_values: List[int],
                 num_simulations: int):
        self.sde = sde
        self.approximation_methods = approximation_methods
        self.dt_grid = dt_grid
        self.p_values = p_values
        self.num_simulations = num_simulations

        self.num_steps_grid = [int(sde.T / dt) for dt in dt_grid]

    def test_different_sde_params(self, param_name: str, param_values: List[float], use_exact_solution: True, visualize=True):
        convergence_rates = {}
        for param_value in param_values:
            print(f"Testing {param_name} = {param_value}")
            setattr(self.sde, param_name, param_value)
            convergence_rates[param_value] = self.calculate_convergence_rates(use_exact_solution, visualize)
        if visualize:
            for method_name, p in convergence_rates[param_values[0]].keys():
                plt.plot(
                    param_values,
                    [convergence_rates[param_value][(method_name, p)] for param_value in param_values],
                    label=f"{method_name} - p = {p}"
                )
            plt.title(f'{"Pseudo " if not use_exact_solution else ""}Approximation Errors for\n {self.sde}')
            plt.xlabel(param_name)
            plt.ylabel('Convergence Rate')
            plt.legend()
            plt.show()
        return convergence_rates

    def calculate_convergence_rates(self, use_exact_solution: True, visualize=True):
        adapted_sde_name = str(self.sde).replace(f"N={self.sde.N}", f"N in {self.num_steps_grid}")
        print(f'SDE: {adapted_sde_name}')

        errors = self._calculate_errors_for_sde(use_exact_solution, visualize)

        convergence_rates = self._calculate_convergence_rates_from_errors(errors)

        return convergence_rates

    def _calculate_errors_for_sde(self, use_exact_solution: True, visualize=True) -> Dict[Tuple[str, int], List[float]]:

        errors = {(method_name, p): [] for method_name in APPROXIMATION_METHODS for p in self.p_values}

        # sample brownian motion with the highest discretization
        highest_res_N = max(self.num_steps_grid) * (2 ** 2)
        highest_res_sde = self.sde.copy()
        highest_res_sde.update_time_discretization(num_steps=highest_res_N)
        brownian_motion = highest_res_sde.sample_brownian_motion(num_simulations=self.num_simulations, seed=SEED)

        if not use_exact_solution:
            print(f"Simulating pseudo exact solutions with N = {highest_res_N}", end="")
            ttime = time()
            pseudo_exact_solutions = {
                method_name: get_approximation_method(method_name)(highest_res_sde, brownian_motion)
                for method_name in APPROXIMATION_METHODS
            }
            print(f" ({time() - ttime:.2f}s)")

        for N in self.num_steps_grid:
            print(f"Running simulations for N = {N}", end="")
            ttime = time()

            self.sde.update_time_discretization(num_steps=N)
            reduced_brownian_motion = brownian_motion[::int(highest_res_N/N)]

            if use_exact_solution:
                exact_solutions = self.sde.exact_solutions(reduced_brownian_motion)
                exact_solutions = {
                    method_name: exact_solutions for method_name in APPROXIMATION_METHODS
                }
            else:
                exact_solutions = {
                    method_name: pseudo_exact_solutions[method_name][::int(highest_res_N/N)]
                    for method_name in APPROXIMATION_METHODS
                }

            for method_name in APPROXIMATION_METHODS:
                approximations = get_approximation_method(method_name)(self.sde, reduced_brownian_motion)
                for p in self.p_values:
                    errors[(method_name, p)].append(
                        self._calculate_lp_error(exact_solutions[method_name], approximations, p=p)
                    )

            print(f" ({time() - ttime:.2f}s)")

        if visualize:
            # plot lp errors
            colors = ['blue', 'orange', 'green']
            linestyles = ['-', '--', ':']
            for i, method_name in enumerate(APPROXIMATION_METHODS):
                for j, p in enumerate(self.p_values):
                    plt.loglog(self.dt_grid, errors[(method_name, p)], label=f"{method_name} - p = {p}",
                               ls=linestyles[j], color=colors[i])
            plt.title(f'{"Pseudo " if not use_exact_solution else ""}Approximation Errors for\n {self.sde}')
            plt.xlabel('$\Delta t$')
            plt.ylabel('Error (e($\Delta t$))')
            plt.legend()
            plt.show()
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
        return error.cpu()

    def _calculate_convergence_rates_from_errors(self, error_values):
        """
        Calculate convergence rate from given errors and time discretizations.
        :param error_values: torch.Tensor: errors of shape (num_dts). $Err(\Delta t)$ above
        :return: float: convergence rate
        """
        print("---")
        convergence_rates = {}
        for method_name in APPROXIMATION_METHODS:
            for p in self.p_values:
                x = torch.log(torch.tensor(self.dt_grid)).reshape(-1, 1)
                y = torch.log(torch.tensor(error_values[(method_name, p)]))
                reg = LinearRegression().fit(x, y)
                convergence_rates[(method_name, p)] = reg.coef_.item()
                print(f"{method_name} convergence rate for p = {p}: {convergence_rates[(method_name, p)]}")
            print("---")
        return convergence_rates


if __name__ == '__main__':

    # SDE = GeometricBrownianMotion(time_horizon=1, num_steps=1, x0=1, mu=2, sigma=1)
    SDE = CoxIngersollRoss(time_horizon=1, num_steps=1, x0=1, a=1, b=0, sigma=2)
    # SDE = OrnsteinUhlenbeck(time_horizon=1, num_steps=1, x0=1, mu=0, theta=1, sigma=1

    calc = ConvergenceRateCalculator(
        sde=SDE,
        approximation_methods=APPROXIMATION_METHODS,
        dt_grid=[2 ** i for i in range(-13, -4)],
        p_values=[1],
        num_simulations=20000,
    )
    # calc.calculate_convergence_rates(use_exact_solution=False, visualize=True)
    calc.test_different_sde_params('a', [0.01, 0.1, 0.25, 0.5, 1, 2, 4], use_exact_solution=False, visualize=True)
