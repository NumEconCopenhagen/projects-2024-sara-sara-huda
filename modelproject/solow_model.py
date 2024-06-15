from types import SimpleNamespace
import numpy as np
from scipy import optimize

class SolowModel:
    
    def __init__(self, do_print=False):
        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()
        self.path = SimpleNamespace()
        self.setup()
        self.allocate()
        if do_print:
            self.print_parameters()
    
    def setup(self):
        par = self.par
        
        # Parameters
        par.alpha = 0.3          # Capital share
        par.beta = 0.95          # Discount factor
        par.delta = 0.05         # Depreciation rate
        par.g = 0.02             # Technological progress
        par.n = 0.01             # Population growth
        par.s = 0.2              # Savings rate
        par.gov_spending = 0.1   # Government spending
        par.Tpath = 100          # Time horizon

        # Initial capital
        par.K_lag_ini = 1.0
    
    def allocate(self):
        par = self.par
        path = self.path
        
        allvarnames = ['K', 'Y', 'C', 'G', 'I']
        for varname in allvarnames:
            path.__dict__[varname] = np.zeros(par.Tpath)
    
    def production_function(self, K):
        par = self.par
        L = 1.0
        A = 1.0
        Y = (K**par.alpha) * (L**(1 - par.alpha)) * A
        return Y
    
    def find_steady_state(self):
        par = self.par
        ss = self.ss
        
        # Solve for steady state capital
        def steady_state_equation(K):
            Y = self.production_function(K)
            C = (1 - par.s) * Y
            I = par.s * Y
            G = par.gov_spending * Y
            return par.s * Y - (par.delta + par.n + par.g) * K - G

        ss.K = optimize.fsolve(steady_state_equation, 1.0)[0]
        ss.Y = self.production_function(ss.K)
        ss.C = (1 - par.s) * ss.Y
        ss.I = par.s * ss.Y
        ss.G = par.gov_spending * ss.Y

    def evaluate_path_errors(self):
        par = self.par
        path = self.path
        ss = self.ss

        C = path.C
        K = path.K
        G = path.G
        Y = path.Y
        I = path.I

        # Calculate errors
        errors = np.zeros(par.Tpath)
        for t in range(par.Tpath):
            errors[t] = (par.s * Y[t] - (par.delta + par.n + par.g) * K[t] - G[t])
        return errors

    def solve(self):
        par = self.par
        ss = self.ss
        path = self.path
        
        path.K[0] = par.K_lag_ini
        for t in range(par.Tpath - 1):
            path.Y[t] = self.production_function(path.K[t])
            path.C[t] = (1 - par.s) * path.Y[t]
            path.I[t] = par.s * path.Y[t]
            path.G[t] = par.gov_spending * path.Y[t]
            path.K[t + 1] = path.K[t] + path.I[t] - par.delta * path.K[t]
        
        path.Y[-1] = self.production_function(path.K[-1])
        path.C[-1] = (1 - par.s) * path.Y[-1]
        path.I[-1] = par.s * path.Y[-1]
        path.G[-1] = par.gov_spending * path.Y[-1]
    
    def apply_shock(self, shock_type, magnitude):
        par = self.par
        if shock_type == 'discount_factor':
            par.beta += magnitude
        elif shock_type == 'savings_rate':
            par.s += magnitude

    def print_parameters(self):
        par = self.par
        print(f"alpha: {par.alpha}")
        print(f"beta: {par.beta}")
        print(f"delta: {par.delta}")
        print(f"g: {par.g}")
        print(f"n: {par.n}")
        print(f"s: {par.s}")
        print(f"gov_spending: {par.gov_spending}")
        print(f"Tpath: {par.Tpath}")
        print(f"K_lag_ini: {par.K_lag_ini}")
