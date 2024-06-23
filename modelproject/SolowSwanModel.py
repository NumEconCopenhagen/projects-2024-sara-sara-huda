import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

class SolowSwanModel:

    def __init__(self, s=0.2, do_print=False):
        self.alpha = 0.3          # Capital share
        self.delta = 0.05         # Depreciation rate
        self.g = 0.02             # Technological progress
        self.n = 0.01             # Population growth
        self.s = s                # Savings rate
        self.K_initial = 1.0      # Initial capital
        self.Tpath = 500          # Time horizon

        if do_print:
            self.print_parameters()

    def print_parameters(self):
        print(f"alpha: {self.alpha}")
        print(f"delta: {self.delta}")
        print(f"g: {self.g}")
        print(f"n: {self.n}")
        print(f"s: {self.s}")
        print(f"K_initial: {self.K_initial}")
        print(f"Tpath: {self.Tpath}")

    def production_function(self, K):
        return K**self.alpha

    def steady_state_equation(self, K):
        return K - (self.s * self.production_function(K) + (1 - self.delta) * K) / ((1 + self.g) * (1 + self.n))

    def find_steady_state(self):
        result = optimize.root_scalar(self.steady_state_equation, bracket=[0.1, 100], method='brentq')
        K_ss = result.root
        Y_ss = self.production_function(K_ss)
        C_ss = (1 - self.s) * Y_ss
        return K_ss, Y_ss, C_ss

    def solve_transition_path(self):
        K_path = np.zeros(self.Tpath)
        Y_path = np.zeros(self.Tpath)
        C_path = np.zeros(self.Tpath)
         # This part of the code involved some difficulties for us due to the iterative computation required for the transition path.
         # After facing difficulties in implementing this on our own, we used ChatGPT to help structure this method correctly.
        
        K_path[0] = self.K_initial
        for t in range(1, self.Tpath):
            Y_path[t-1] = self.production_function(K_path[t-1])
            C_path[t-1] = (1 - self.s) * Y_path[t-1]
            K_path[t] = K_path[t-1] + self.s * Y_path[t-1] - (self.delta + self.n + self.g) * K_path[t-1]

        Y_path[-1] = self.production_function(K_path[-1])
        C_path[-1] = (1 - self.s) * Y_path[-1]
        return K_path, Y_path, C_path

    def plot_results(self, K_path, Y_path, C_path, K_ss, Y_ss, C_ss):
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(K_path, label='Physical Capital (K)')
        plt.axhline(y=K_ss, color='r', linestyle='--', label='Steady State K')
        plt.xlabel('Time')
        plt.ylabel('Physical Capital')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(Y_path, label='Output (Y)')
        plt.axhline(y=Y_ss, color='r', linestyle='--', label='Steady State Y')
        plt.xlabel('Time')
        plt.ylabel('Output')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(C_path, label='Consumption (C)')
        plt.axhline(y=C_ss, color='r', linestyle='--', label='Steady State C')
        plt.xlabel('Time')
        plt.ylabel('Consumption')
        plt.legend()

        plt.tight_layout()
        plt.show()
