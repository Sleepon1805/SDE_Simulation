import math
import torch
import matplotlib.pyplot as plt
from time import time
from typing import List, Dict, Tuple
from sklearn.linear_model import LinearRegression
from labellines import labelLines
from pprint import pprint

from sde_lib import SDE, GeometricBrownianMotion, OrnsteinUhlenbeck, CoxIngersollRoss
from approximation_methods import get_approximation_method
from utils import SEED

plt.rcParams['figure.figsize'] = (10, 6)
DEFAULT_BATCH_SIZE = 5000  # 20000 (num_simulations) x 32768 (timesteps) still pass on my Radeon RX 6700 XT


class ConvergenceRateCalculator:
    def __init__(self, sde: SDE, approximation_methods: List[str], dt_grid: List[float], p_values: List[int],
                 num_simulations: int, batch_size: int = DEFAULT_BATCH_SIZE):
        self.sde = sde
        self.approximation_methods = approximation_methods
        self.dt_grid = dt_grid
        self.p_values = p_values
        self.num_simulations = num_simulations

        self.num_steps_grid = [int(sde.T / dt) for dt in dt_grid]

        # for gpu calculations
        self.batch_size = min(batch_size, num_simulations)

        # for plotting
        self.colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        self.linestyles = ['-', '--', ':']
        self.seed = SEED

    def test_different_sde_params(self, param_name: str, param_values: List[float],
                                  use_exact_solution: True, visualize=True):
        convergence_rates = {}
        for param_value in param_values:
            print(f"Testing {param_name} = {param_value}")
            setattr(self.sde, param_name, param_value)
            convergence_rates[param_value] = self.calculate_convergence_rates(use_exact_solution, visualize)

        if visualize:
            self.visualize_convergence_rates(param_name, param_values, convergence_rates)

        print(f"Convergence rates for {param_name} = {param_values}:")
        pprint(convergence_rates)
        return convergence_rates

    def calculate_convergence_rates(self, use_exact_solution: True, visualize=True):
        print(f"SDE: {self.sde.to_str(hide='N')}")

        errors = self.calculate_errors_batched(use_exact_solution)

        convergence_rates = self._calculate_convergence_rates_from_errors(errors)

        if visualize:
            self.visualize_errors(errors)
        return convergence_rates

    def calculate_errors_batched(self, use_exact_solution: bool) -> Dict[Tuple[str, int, int], float]:
        errors = []
        assert self.num_simulations % self.batch_size == 0, "float number of batches not implemented yet"
        num_batches = self.num_simulations // self.batch_size  # TODO: automatic batch size
        ttime = time()
        for batch_num in range(num_batches):
            print(f'Batch {batch_num + 1}/{num_batches}')
            errors.append(self._calculate_errors(self.batch_size, use_exact_solution))
        errors = {key: sum([error[key] for error in errors]) / num_batches for key in errors[0].keys()}
        print(f"Total simulation time: {time() - ttime:.2f}s")
        return errors

    def _calculate_errors(self, num_simulations: int, use_exact_solution: bool) -> Dict[Tuple[str, int, int], float]:
        # sample brownian motion with the highest discretization
        highest_res_N = max(self.num_steps_grid) * 2
        highest_res_sde = self.sde.copy()
        highest_res_sde.update_time_discretization(num_steps=highest_res_N)
        brownian_motion = highest_res_sde.sample_brownian_motion(num_simulations=num_simulations, seed=self.seed)
        self.seed *= 2

        # calculate exact solutions as approximations with higher discretization
        approximations = {}
        if use_exact_solution:
            print(f"Simulating exact solutions with N = {highest_res_N}", end="")
            ttime = time()
            exact_solutions = highest_res_sde.exact_solutions(brownian_motion)
            print(f" ({time() - ttime:.2f}s)")
        else:
            print(f"Running simulations for N = {highest_res_N} (for pseudo exact solutions)", end="")
            ttime = time()
            exact_solutions = None
            for method_name in self.approximation_methods:
                approximations[(method_name, highest_res_N)] = get_approximation_method(method_name)(
                    highest_res_sde, brownian_motion)
            print(f" ({time() - ttime:.2f}s)")

        # run simulations for different time discretizations
        print(f"Running simulations for N in {self.num_steps_grid}:")
        for N in self.num_steps_grid:
            ttime = time()
            self.sde.update_time_discretization(num_steps=N)
            for method_name in self.approximation_methods:
                reduced_brownian_motion = brownian_motion[::int(highest_res_N/N)]
                approximations[(method_name, N)] = get_approximation_method(method_name)(
                    self.sde, reduced_brownian_motion)
            print(f"{time() - ttime:.2f}s + ", end="")

        # calculate errors
        errors = {}
        for method_name in self.approximation_methods:
            for N in self.num_steps_grid:
                if not use_exact_solution:
                    reduced_exact_solutions = approximations[(method_name, 2 * N)][::2]
                else:
                    reduced_exact_solutions = exact_solutions[::int(highest_res_N/N)]
                for p in self.p_values:
                    errors[(method_name, p, N)] = self._calculate_lp_error(
                        reduced_exact_solutions, approximations[(method_name, N)], p=p)
        print("")
        return errors

    @staticmethod
    def _calculate_lp_error(solutions, calc_approximations, p):
        assert p >= 1, "p \in [1, \infty)"
        error = torch.max(torch.mean(torch.abs(solutions - calc_approximations) ** p, dim=1) ** (1/p))
        return error.cpu().item()

    def _calculate_convergence_rates_from_errors(self, error_values: Dict[Tuple[str, int, int], float]
                                                 ) -> Dict[Tuple[str, int], float]:
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

                # minN, maxN = min(self.num_steps_grid), max(self.num_steps_grid)
                # error_minN, error_maxN = error_values[(method_name, p, minN)], error_values[(method_name, p, maxN)]
                # convergence_rates[(method_name, p)] = ((math.log(error_minN) - math.log(error_maxN))
                #                                        / math.log(maxN / minN))

                print(f"{method_name} convergence rate for p = {p}: {convergence_rates[(method_name, p)]}")
            print("---")
        return convergence_rates

    def visualize_errors(self, errors: Dict[Tuple[str, int, int], float]):
        # plot lp errors
        for i, method_name in enumerate(self.approximation_methods):
            for j, p in enumerate(self.p_values):
                plt.loglog(self.dt_grid, [errors[(method_name, p, N)] for N in self.num_steps_grid],
                           label=f"{method_name}", color=self.colors[i], ls=self.linestyles[j])
        labelLines(plt.gca().get_lines())
        # plt.legend()
        plt.title(f'Approximation Errors for\n {self.sde}')
        plt.xlabel('$\Delta t$')
        plt.ylabel('Error (e($\Delta t$))')
        plt.show()

    def visualize_convergence_rates(self, param_name: str, param_values: list,
                                    convergence_rates: Dict[float, Dict[Tuple[str, int], float]]):
        x_values = param_values
        plt.xlabel(param_name)
        # show delta values for CIR process
        if isinstance(self.sde, CoxIngersollRoss):
            if param_name == 'a':
                x_values = [4 * a / (self.sde.sigma ** 2) for a in param_values]
                plt.xlabel("delta")
            elif param_name == 'sigma':
                x_values = [4 * self.sde.a / (sigma ** 2) for sigma in param_values]
                plt.xlabel("delta")

        for i, method_name in enumerate(self.approximation_methods):
            for j, p in enumerate(self.p_values):
                plt.plot(
                    x_values,
                    [convergence_rates[param_value][(method_name, p)] for param_value in param_values],
                    label=f"{method_name}", color=self.colors[i], ls=self.linestyles[j]
                )
        labelLines(plt.gca().get_lines())
        # plt.legend()
        plt.title(f"Approximation Errors for\n {self.sde.to_str(hide=['N', param_name])}\n N = {self.num_steps_grid}")
        plt.ylabel('Convergence Rate')
        plt.show()


if __name__ == '__main__':
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

    # SDE = GeometricBrownianMotion(time_horizon=1, num_steps=1, x0=1, mu=2, sigma=1)
    SDE = CoxIngersollRoss(time_horizon=1, num_steps=1, x0=1, a=1, b=1, sigma=2)
    # SDE = OrnsteinUhlenbeck(time_horizon=1, num_steps=1, x0=1, mu=0, theta=1, sigma=1

    calc = ConvergenceRateCalculator(
        sde=SDE,
        approximation_methods=APPROXIMATION_METHODS,
        dt_grid=[2 ** i for i in range(-13, -2)],
        p_values=[1],
        num_simulations=10000,
    )
    # calc.calculate_convergence_rates(use_exact_solution=False, visualize=True)
    calc.test_different_sde_params('a', [0.25, 0.5, 1, 2, 3, 4],
                                   use_exact_solution=False, visualize=True)
