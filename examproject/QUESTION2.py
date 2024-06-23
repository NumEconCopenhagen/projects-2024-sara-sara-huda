import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

def initialise_parameters():
    par = SimpleNamespace()
    par.J = 3
    par.N = 10
    par.K = 10000
    par.F = np.arange(1, par.N + 1)
    par.sigma = 2
    par.v = np.array([1, 2, 3])
    par.c = 1
    return par

def simulate_error_term(par):
    np.random.seed(42)
    return np.random.normal(0, par.sigma, (par.J, par.K))

def calculate_utilities(par, epsilon):
    expected_utility = par.v + np.mean(epsilon, axis=1)
    realised_utility = par.v[:, np.newaxis] + epsilon
    average_realised_utility = np.mean(realised_utility, axis=1)
    return expected_utility, realised_utility, average_realised_utility

def plot_histogram(realised_utility, par):
    fig, ax = plt.subplots(figsize=(6, 4))
    for j in range(par.J):
        ax.hist(realised_utility[j, :], bins=50, alpha=0.6, label=f'Career {j + 1}')
    ax.set_xlabel('Realised Utility')
    ax.set_ylabel('Frequency')
    ax.set_title('Figure 1(Q1): Distribution of Realised Utilities for Each Career')
    ax.legend()
    plt.show()

def simulate_career_choice(par):
    chosen_careers = np.zeros((par.N, par.K), dtype=int)
    expectations_before = np.zeros((par.N, par.K))
    realised_utilities = np.zeros((par.N, par.K))

    for k in range(par.K):
        for i in range(par.N):
            Fi = par.F[i]
            epsilon_fj = np.random.normal(0, par.sigma, (par.J, Fi))
            epsilon_ij = np.random.normal(0, par.sigma, par.J)
            expected_utility_before = par.v[:, np.newaxis] + epsilon_fj
            expected_utility_before_mean = np.mean(expected_utility_before, axis=1)
            chosen_career = np.argmax(expected_utility_before_mean)
            chosen_careers[i, k] = chosen_career
            expectations_before[i, k] = expected_utility_before_mean[chosen_career]
            realised_utilities[i, k] = par.v[chosen_career] + epsilon_ij[chosen_career]

    return chosen_careers, expectations_before, realised_utilities

def calculate_results(par, chosen_careers, expectations_before, realised_utilities):
    career_share = np.zeros((par.N, par.J))
    average_subjective_utility = np.zeros(par.N)
    average_realised_utility = np.zeros(par.N)

    for i in range(par.N):
        for j in range(par.J):
            career_share[i, j] = np.mean(chosen_careers[i] == j)
        average_subjective_utility[i] = np.mean(expectations_before[i])
        average_realised_utility[i] = np.mean(realised_utilities[i])

    return career_share, average_subjective_utility, average_realised_utility

def plot_results(par, career_share, average_subjective_utility, average_realised_utility):
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))

    for j in range(par.J):
        ax[0].plot(par.F, career_share[:, j], label=f'Career {j + 1}')
    ax[0].set_xlabel('Graduate (i)')
    ax[0].set_ylabel('Share Choosing Career')
    ax[0].legend()
    ax[0].set_title('Figure 1(Q2): Share of the Graduates Choosing Each Career')

    ax[1].plot(par.F, average_subjective_utility, marker='o')
    ax[1].set_xlabel('Graduate (i)')
    ax[1].set_ylabel('Average Subjective Expected Utility')
    ax[1].set_title('Figure 2(Q2): Average Subjective Expected Utility')

    ax[2].plot(par.F, average_realised_utility, marker='o')
    ax[2].set_xlabel('Graduate (i)')
    ax[2].set_ylabel('Average Ex Post Realised Utility')
    ax[2].set_title('Figure 3(Q2): Average Ex Post Realised Utility')

    plt.tight_layout()
    plt.show()

def simulate_new_career_choice(par, chosen_careers, expectations_before, realised_utilities):
    new_chosen_careers = np.zeros((par.N, par.K), dtype=int)
    new_expectations_before = np.zeros((par.N, par.K))
    new_realised_utilities = np.zeros((par.N, par.K))
    decisions_to_switch = np.zeros((par.N, par.K), dtype=bool)

    for k in range(par.K):
        for i in range(par.N):
            Fi = par.F[i]
            original_career = chosen_careers[i, k]
            new_expected_utility_before = np.zeros(par.J)
            for j in range(par.J):
                if j == original_career:
                    new_expected_utility_before[j] = realised_utilities[i, k]
                else:
                    new_expected_utility_before[j] = expectations_before[i, k] - par.c
            new_chosen_career = np.argmax(new_expected_utility_before)
            decisions_to_switch[i, k] = (new_chosen_career != original_career)
            new_chosen_careers[i, k] = new_chosen_career
            new_expectations_before[i, k] = new_expected_utility_before[new_chosen_career]
            new_realised_utilities[i, k] = realised_utilities[i, k] if new_chosen_career == original_career else par.v[new_chosen_career] + np.random.normal(0, par.sigma) - par.c

    return new_chosen_careers, new_expectations_before, new_realised_utilities, decisions_to_switch

def calculate_new_results(par, chosen_careers, decisions_to_switch, new_expectations_before, new_realised_utilities):
    new_average_subjective_utility = np.zeros(par.N)
    new_average_realised_utility = np.zeros(par.N)
    switch_share = np.zeros((par.N, par.J))

    for i in range(par.N):
        new_average_subjective_utility[i] = np.mean(new_expectations_before[i])
        new_average_realised_utility[i] = np.mean(new_realised_utilities[i])
        for j in range(par.J):
            switch_share[i, j] = np.mean(decisions_to_switch[i] & (chosen_careers[i] == j))

    return new_average_subjective_utility, new_average_realised_utility, switch_share

def plot_new_results(par, switch_share, new_average_subjective_utility, new_average_realised_utility):
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))

    for j in range(par.J):
        ax[0].plot(par.F, switch_share[:, j], label=f'Original Career {j + 1}')
    ax[0].set_xlabel('Graduate (i)')
    ax[0].set_ylabel('Share Choosing to Switch')
    ax[0].legend()
    ax[0].set_title('Figure 1(Q3): Share of Graduates Who Choose to Switch Careers')

    ax[1].plot(par.F, new_average_subjective_utility, marker='o')
    ax[1].set_xlabel('Graduate (i)')
    ax[1].set_ylabel('New Average Subjective Expected Utility')
    ax[1].set_title('Figure 2(Q3): New Average Subjective Expected Utility')

    ax[2].plot(par.F, new_average_realised_utility, marker='o')
    ax[2].set_xlabel('Graduate (i)')
    ax[2].set_ylabel('New Average Ex Post Realised Utility')
    ax[2].set_title('Figure 3(Q3): New Average Ex Post Realised Utility')

    plt.tight_layout()
    plt.show()

def main():
    par = initialise_parameters()
    epsilon = simulate_error_term(par)
    
    expected_utility, realised_utility, average_realised_utility = calculate_utilities(par, epsilon)
    
    for j in range(par.J):
        print(f"Career {j + 1}:")
        print(f"Expected Utility: {expected_utility[j]:.4f}")
        print(f"Average Realised Utility: {average_realised_utility[j]:.4f}")
    
    plot_histogram(realised_utility, par)
    
    chosen_careers, expectations_before, realised_utilities = simulate_career_choice(par)
    
    career_share, average_subjective_utility, average_realised_utility = calculate_results(par, chosen_careers, expectations_before, realised_utilities)
    
    plot_results(par, career_share, average_subjective_utility, average_realised_utility)
    
    new_chosen_careers, new_expectations_before, new_realised_utilities, decisions_to_switch = simulate_new_career_choice(par, chosen_careers, expectations_before, realised_utilities)
    
    new_average_subjective_utility, new_average_realised_utility, switch_share = calculate_new_results(par, chosen_careers, decisions_to_switch, new_expectations_before, new_realised_utilities)
    
    plot_new_results(par, switch_share, new_average_subjective_utility, new_average_realised_utility)
