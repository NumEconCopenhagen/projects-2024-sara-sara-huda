import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

class ExchangeEconomyClass:
    def __init__(self, alpha, beta, endowment_A, endowment_B):
        self.alpha = alpha
        self.beta = beta
        self.endowment_A = endowment_A
        self.endowment_B = endowment_B

    def utility_A(self, x1, x2):
        return (x1**self.alpha) * (x2**(1-self.alpha))

    def utility_B(self, x1, x2):
        return (x1**self.beta) * (x2**(1-self.beta))

    def x_A1_star(self, p1):
        return self.alpha * ((self.endowment_A[0] * p1 + self.endowment_A[1]) / p1)
    
    def x_A2_star(self, p1):
        return (1 - self.alpha) * (self.endowment_A[0] * p1 + self.endowment_A[1])
    
    def x_B1_star(self, p1):
        return self.beta * ((self.endowment_B[0] * p1 + self.endowment_B[1]) / p1)
    
    def x_B2_star(self, p1):
        return (1 - self.beta) * (p1 * self.endowment_B[0] + self.endowment_B[1])

    def market_clearing_price(self):
        def market_clearing_condition(p1):
            x_A1 = self.x_A1_star(p1)
            x_A2 = self.x_A2_star(p1)
            x_B1 = self.x_B1_star(p1)
            x_B2 = self.x_B2_star(p1)
            return (x_A1 + x_B1 - self.endowment_A[0] - self.endowment_B[0],
                    x_A2 + x_B2 - self.endowment_A[1] - self.endowment_B[1])

        result = minimize(lambda p1: np.linalg.norm(market_clearing_condition(p1)), x0=1.0, bounds=[(0.01, None)])
        return result.x[0]
    
    def pareto_improvements(self):
        initial_utility_A = self.utility_A(*self.endowment_A)
        initial_utility_B = self.utility_B(*self.endowment_B)

        pareto_improvements = []
        x1_range = np.linspace(0, 1, 75)
        x2_range = np.linspace(0, 1, 75)

        for x1 in x1_range:
            for x2 in x2_range:
                wA1_remaining = 1 - x1
                wA2_remaining = 1 - x2
                x1B = 1 - x1
                x2B = 1 - x2

                current_utility_A = self.utility_A(x1, x2)
                current_utility_B = self.utility_B(x1B, x2B)

                if current_utility_A >= initial_utility_A and current_utility_B >= initial_utility_B:
                    pareto_improvements.append((x1, x2))
        
        return pareto_improvements

    def plot_endowment(self):
        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        w1A, w2A = self.endowment_A
        ax_A.scatter(w1A, w2A, marker='s', color='black', label='Endowment')

        w1bar = 1.0
        w2bar = 1.0
        ax_A.plot([0, w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, w1bar], [w2bar, w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, w2bar], lw=2, color='black')
        ax_A.plot([w1bar, w1bar], [0, w2bar], lw=2, color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.6, 1.0))

        pareto_improvements = self.pareto_improvements()
        pareto_x1, pareto_x2 = zip(*pareto_improvements)
        ax_A.scatter(pareto_x1, pareto_x2, alpha=0.5, label='Pareto improvements', color='blue')

        plt.show()

    def plot_market_clearing_errors(self):
        p1_values = np.linspace(0.5, 2.5, 76)
        errors_1 = []
        errors_2 = []

        for p1 in p1_values:
            x_A1 = self.x_A1_star(p1)
            x_A2 = self.x_A2_star(p1)
            x_B1 = self.x_B1_star(p1)
            x_B2 = self.x_B2_star(p1)
            
            epsilon_1 = x_A1 + x_B1 - self.endowment_A[0] - self.endowment_B[0]
            epsilon_2 = x_A2 + x_B2 - self.endowment_A[1] - self.endowment_B[1]
            
            errors_1.append(epsilon_1)
            errors_2.append(epsilon_2)

        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.plot(p1_values, errors_1, label=r'$\epsilon_1$')
        plt.xlabel('$p_1$')
        plt.ylabel('Error')
        plt.title('Market Clearing Error $\epsilon_1$')
        plt.axhline(0, color='red', linestyle='--')
        plt.axvline(0.944, color='red', linestyle='--')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(p1_values, errors_2, label=r'$\epsilon_2$')
        plt.xlabel('$p_1$')
        plt.ylabel('Error')
        plt.title('Market Clearing Error $\epsilon_2$')
        plt.axhline(0, color='red', linestyle='--')
        plt.axvline(0.944, color='red', linestyle='--')
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    alpha = 1/3
    beta = 2/3
    endowment_A = [0.8, 0.3]
    endowment_B = [1 - endowment_A[0], 1 - endowment_A[1]]

    economy = ExchangeEconomyClass(alpha, beta, endowment_A, endowment_B)
    
    # Market clearing price
    p1_clearing = economy.market_clearing_price()
    print(f'Market clearing price (p1): {p1_clearing:.3f}')

    # Plot endowment and Pareto improvements
    economy.plot_endowment()

    # Plot market clearing errors
    economy.plot_market_clearing_errors()

