from scipy import optimize

class ExchangeEconomyClass:
    def __init__(self, alpha, beta, endowment_A):
        self.alpha = alpha
        self.beta = beta
        self.endowment_A = endowment_A
        self.endowment_B = [1 - e for e in endowment_A]

    def utility_A(self, x1, x2):
        return x1**self.alpha * x2**(1 - self.alpha)

    def utility_B(self, x1, x2):
        return x1**self.beta * x2**(1 - self.beta)

    def demand_A_x1(self, p1, p2):
        return self.alpha * (p1*self.endowment_A[0] + p2*self.endowment_A[1]) / p1

    def demand_A_x2(self, p1, p2):
        return (1 - self.alpha) * (p1*self.endowment_A[0] + p2*self.endowment_A[1]) / p2

    def demand_B_x1(self, p1, p2):
        return self.beta * (p1*self.endowment_B[0] + p2*self.endowment_B[1]) / p1

    def demand_B_x2(self, p1, p2):
        return (1 - self.beta) * (p1*self.endowment_B[0] + p2*self.endowment_B[1]) / p2

    # Market clearing price
    def market_clearing_price(self):
        def excess_demand_x1(p1):
            total_demand_x1 = self.demand_A_x1(p1, 1) + self.demand_B_x1(p1, 1)
            total_endowment_x1 = self.endowment_A[0] + self.endowment_B[0]
            return total_demand_x1 - total_endowment_x1

        p1_clearing = optimize.brentq(excess_demand_x1, 0.01, 10)
        return p1_clearing

    # Allocation if only prices in P1 can be chosen
    def allocation_prices_in_P1(self, P1):
        def objective(p1):
            x1 = self.demand_A_x1(p1, 1)
            x2 = self.demand_A_x2(p1, 1)
            return -self.utility_A(x1, x2)

        res = optimize.minimize_scalar(objective, bounds=(min(P1), max(P1)), method='bounded')
        p1_optimal = res.x
        x1_A = self.demand_A_x1(p1_optimal, 1)
        x2_A = self.demand_A_x2(p1_optimal, 1)
        x1_B = self.demand_B_x1(p1_optimal, 1)
        x2_B = self.demand_B_x2(p1_optimal, 1)
        return p1_optimal, (x1_A, x2_A), (x1_B, x2_B)

    # Allocation if any positive price can be chosen 
    def allocation_any_positive_price(self):
        def objective(p1):
            x1 = self.demand_A_x1(p1, 1)
            x2 = self.demand_A_x2(p1, 1)
            return -self.utility_A(x1, x2)

        res = optimize.minimize_scalar(objective, bounds=(0.01, 10), method='bounded')
        p1_optimal = res.x
        x1_A = self.demand_A_x1(p1_optimal, 1)
        x2_A = self.demand_A_x2(p1_optimal, 1)
        x1_B = self.demand_B_x1(p1_optimal, 1)
        x2_B = self.demand_B_x2(p1_optimal, 1)
        return p1_optimal, (x1_A, x2_A), (x1_B, x2_B)

if __name__ == '__main__':
    # Parameters and preferences
    alpha = 1/3
    beta = 2/3
    endowment_A = [0.8, 0.3]

    # Initialize the economy 
    economy = ExchangeEconomyClass(alpha, beta, endowment_A)

    # Market clearing price 
    p1_clearing = economy.market_clearing_price()
    print(f'Market clearing price (p1): {p1_clearing}')

    # Assume a set of prices P1 for question 4a
    P1 = [0.5, 1.5]
    p1_optimal_4a, allocation_A_4a, allocation_B_4a = economy.allocation_prices_in_P1(P1)
    print(f'Optimal price and allocation in P1: {p1_optimal_4a}, {allocation_A_4a}, {allocation_B_4a}')

    # Allocation for question 4b
    p1_optimal_4b, allocation_A_4b, allocation_B_4b = economy.allocation_any_positive_price()
    print(f'Optimal price and allocation for any positive price: {p1_optimal_4b}, {allocation_A_4b}, {allocation_B_4b}')
