
import numpy as np
import pandas as pd

class RBCModel:
    def __init__(self, parameters):
        """Initialize the RBC Model with given parameters."""
        self.parameters = parameters
        self.ss = None
        self.a = None
        self.b = None
        self.f = None
        self.p = None

    def equations(self, variables_forward, variables_current):
        """Define equilibrium equations of the RBC model."""
        p = self.parameters
        fwd = variables_forward
        cur = variables_current

        # Household Euler equation
        euler_eqn = (p['beta'] * fwd['c']**(-p['sigma']) * (p['alpha'] * fwd['k']**(p['alpha'] - 1) * fwd['a'] + 1 - p['delta']) - cur['c']**(-p['sigma']))

        # Production function
        production_function = cur['a'] * cur['k']**p['alpha'] - cur['y']

        # Capital evolution
        capital_evolution = fwd['k'] - (1 - p['delta']) * cur['k'] - cur['i']

        # Market clearing condition
        market_clearing = cur['y'] - cur['c'] - cur['i']

        # Exogenous technology process
        technology_proc = cur['a']**p['rhoa'] * np.exp(fwd['e_a']) - fwd['a']

        return np.array([euler_eqn, production_function, capital_evolution, market_clearing, technology_proc])

    def compute_ss(self, guess):
        """Compute the steady-state values for the RBC model numerically."""
        p = self.parameters
        a, k, c, y, i = guess

        k = ((p['alpha'] * a) / (1 / p['beta'] + p['delta'] - 1))**(1 / (1 - p['alpha']))
        y = a * k**p['alpha']
        c = y - p['delta'] * k
        i = p['delta'] * k

        self.ss = pd.Series([a, k, c, y, i], index=['a', 'k', 'c', 'y', 'i'])
        return self.ss

    def log_linear_approximation(self):
        """Perform log-linear approximation around the steady state."""
        p = self.parameters
        ss = self.ss
        a, k, c, y, i = ss['a'], ss['k'], ss['c'], ss['y'], ss['i']

        a_matrix = np.zeros((2, 2))
        b_matrix = np.zeros((2, 1))

        # Dummy matrices representing the linear approximation
        a_matrix[0, 0] = p['beta'] * p['alpha']
        b_matrix[0, 0] = 0.5

        self.a = a_matrix
        self.b = b_matrix

    def solve_klein(self):
        """Solve the model using a Klein method or similar (example implementation)."""
        self.f = np.array([[0.5, 0.5], [0.1, 0.9]])
        self.p = np.array([[0.8, 0.2], [0.3, 0.7]])

    def simulate_model(self, T=100, seed=None):
        """Simulate the RBC model for T periods with an optional random seed."""
        if seed is not None:
            np.random.seed(seed)

        sim_data = np.zeros((T, 2))
        state = np.array([0.5, 0.5])

        for t in range(T):
            state = self.f @ state + self.p @ np.random.randn(2)
            sim_data[t] = state

        return pd.DataFrame(sim_data, columns=['k', 'y'])
