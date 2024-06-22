import numpy as np
from types import SimpleNamespace
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt

# Parameters 
par = SimpleNamespace()
par.A = 1.0
par.gamma = 0.5
par.alpha = 0.3
par.nu = 1.0
par.epsilon = 2.0
par.tau = 0.0
par.T = 0.0
par.kappa = 0.1

w = 1.0  # Numeraire

def labor_demand(w, p, A, gamma):
    return (p * A * gamma / w) ** (1 / (1 - gamma))

def production(A, labor, gamma):
    return A * labor ** gamma

def profit(w, p, A, gamma):
    labor = labor_demand(w, p, A, gamma)
    output = production(A, labor, gamma)
    return p * output - w * labor

def utility_maximization(l, w, p1, p2, alpha, nu, epsilon, T, pi1, pi2, tau):
    income = w * l + T + pi1 + pi2
    c1 = alpha * income / p1
    c2 = (1 - alpha) * income / (p2 + tau)
    if c1 <= 0 or c2 <= 0:
        return -np.inf
    utility = np.log(c1 ** alpha * c2 ** (1 - alpha)) - nu * l ** (1 + epsilon) / (1 + epsilon)
    return utility

def market_clearing_conditions(p1, p2, w):
    l1 = labor_demand(w, p1, par.A, par.gamma)
    y1 = production(par.A, l1, par.gamma)
    pi1 = profit(w, p1, par.A, par.gamma)

    l2 = labor_demand(w, p2, par.A, par.gamma)
    y2 = production(par.A, l2, par.gamma)
    pi2 = profit(w, p2, par.A, par.gamma)

    l_star = fsolve(lambda l: utility_maximization(l, w, p1, p2, par.alpha, par.nu, par.epsilon, par.T, pi1, pi2, par.tau), 1.0)[0]
    c1_star = par.alpha * (w * l_star + par.T + pi1 + pi2) / p1
    c2_star = (1 - par.alpha) * (w * l_star + par.T + pi1 + pi2) / (p2 + par.tau)

    labor_market = l_star - (l1 + l2)
    goods_market_1 = c1_star - y1
    goods_market_2 = c2_star - y2

    return labor_market, goods_market_1, goods_market_2

def equilibrium_conditions(prices):
    p1, p2 = prices
    w = 1.0

    l1 = labor_demand(w, p1, par.A, par.gamma)
    y1 = production(par.A, l1, par.gamma)
    pi1 = profit(w, p1, par.A, par.gamma)

    l2 = labor_demand(w, p2, par.A, par.gamma)
    y2 = production(par.A, l2, par.gamma)
    pi2 = profit(w, p2, par.A, par.gamma)

    l_star = fsolve(lambda l: utility_maximization(l, w, p1, p2, par.alpha, par.nu, par.epsilon, par.T, pi1, pi2, par.tau), 1.0)[0]
    c1_star = par.alpha * (w * l_star + par.T + pi1 + pi2) / p1
    c2_star = (1 - par.alpha) * (w * l_star + par.T + pi1 + pi2) / (p2 + par.tau)

    labor_market = l_star - (l1 + l2)
    goods_market_1 = c1_star - y1

    return [labor_market, goods_market_1]

def compute_equilibrium_prices(tau, T):
    def equilibrium_conditions(prices):
        p1, p2 = prices
        w = 1.0

        l1 = labor_demand(w, p1, par.A, par.gamma)
        y1 = production(par.A, l1, par.gamma)
        pi1 = profit(w, p1, par.A, par.gamma)

        l2 = labor_demand(w, p2, par.A, par.gamma)
        y2 = production(par.A, l2, par.gamma)
        pi2 = profit(w, p2, par.A, par.gamma)

        l_star = fsolve(lambda l: utility_maximization(l, w, p1, p2, par.alpha, par.nu, par.epsilon, T, pi1, pi2, tau), 1.0)[0]
        c1_star = par.alpha * (w * l_star + T + pi1 + pi2) / p1
        c2_star = (1 - par.alpha) * (w * l_star + T + pi1 + pi2) / (p2 + tau)

        labor_market = l_star - (l1 + l2)
        goods_market_1 = c1_star - y1

        return [labor_market, goods_market_1]

    initial_guess = [1.0, 1.0]
    equilibrium_prices = fsolve(equilibrium_conditions, initial_guess)
    return equilibrium_prices

def social_welfare(params):
    tau, T = params
    p1, p2 = compute_equilibrium_prices(tau, T)
    w = 1.0

    l1 = labor_demand(w, p1, par.A, par.gamma)
    y1 = production(par.A, l1, par.gamma)
    pi1 = profit(w, p1, par.A, par.gamma)

    l2 = labor_demand(w, p2, par.A, par.gamma)
    y2 = production(par.A, l2, par.gamma)
    pi2 = profit(w, p2, par.A, par.gamma)

    l_star = fsolve(lambda l: utility_maximization(l, w, p1, p2, par.alpha, par.nu, par.epsilon, T, pi1, pi2, tau), 1.0)[0]
    c1_star = par.alpha * (w * l_star + T + pi1 + pi2) / p1
    c2_star = (1 - par.alpha) * (w * l_star + T + pi1 + pi2) / (p2 + tau)

    if c1_star <= 0 or c2_star <= 0 or l_star <= 0:
        return np.inf

    U = np.log(c1_star ** par.alpha * c2_star ** (1 - par.alpha)) - par.nu * l_star ** (1 + par.epsilon) / (1 + par.epsilon)

    SWF = U - par.kappa * y2

    return -SWF
