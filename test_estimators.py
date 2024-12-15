from network_graph import *
from estimators import *

def test_estimators_weak(graph_model_class, init_num_nodes=10000, alpha=2.2, p_share=0.5, seed=42, sample_sizes=[10, 20, 30, 40, 50]):
    """
    Compare the vanilla and friendship paradox-based estimators and plot their absolute errors.
    """
    graph_model = graph_model_class(init_num_nodes=init_num_nodes, alpha=alpha, p_share=p_share, seed=seed)

    vanilla_errors = []
    friendship_errors = []
    hybrid_errors = []
    fotouhi_errors = []

    for n_samples in sample_sizes:
        vanilla_estimate = vanilla_estimator(graph_model, n_samples)
        friendship_estimate = friendship_paradox_estimator(graph_model, n_samples)
        hybrid_estimate = hybrid_estimator(graph_model, n_samples)
        fotouhi_estimate = fotouhi_estimator(graph_model, n_samples)

        # Absolute error in percentage
        vanilla_error = abs(vanilla_estimate - graph_model.true_average_exposure) / graph_model.true_average_exposure * 100
        friendship_error = abs(friendship_estimate - graph_model.true_average_exposure) / graph_model.true_average_exposure * 100
        hybrid_error = abs(hybrid_estimate - graph_model.true_average_exposure) / graph_model.true_average_exposure * 100
        fotouhi_error = abs(fotouhi_estimate - graph_model.true_average_exposure) / graph_model.true_average_exposure * 100

        vanilla_errors.append(vanilla_error)
        friendship_errors.append(friendship_error)
        hybrid_errors.append(hybrid_error)
        fotouhi_errors.append(fotouhi_error)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, vanilla_errors, label="Vanilla Estimator", marker="o", color="red")
    plt.plot(sample_sizes, friendship_errors, label="Friendship Paradox Estimator", marker="s", color="green")
    plt.plot(sample_sizes, hybrid_errors, label="Hybrid Estimator", marker="^", color="blue")
    plt.plot(sample_sizes, fotouhi_errors, label="Fotouhi Estimator", marker="*", color="purple")
    plt.xlabel("Sample Size (n)")
    plt.ylabel("Absolute Error (%)")
    plt.title("Comparison of Estimators")
    plt.legend()
    plt.grid()
    plt.savefig("estimator_performance_weak.png")

def test_estimators_strong(graph_model_class, init_num_nodes=10000, alpha=2.2, pshare_values=[0.005, 0.015, 0.025, 0.5], 
                           sample_sizes=[10, 20, 30, 40, 50], monte_carlo_iterations=100, seed=None):
    """
    Tests and plots the performance of vanilla and friendship paradox-based estimators
    against different Pshare values using a Monte Carlo average of 5,000 iterations. 
    This involves running the estimators 'monte_carlo_iterations' times per configuration
    (e.g., for each combination of pshare and sample size to compute a more robust average
    error.

    Parameters:
    - graph_model_class: The class used to create the graph model (e.g., GraphModel).
    - init_num_nodes: Number of inital nodes in the graph before connectivity assurance.
    - alpha: Power-law exponent for the graph's degree distribution.
    - pshare_values: List of Pshare values to test.
    - sample_sizes: List of sample sizes for the estimators.
    - monte_carlo_iterations: Number of Monte Carlo iterations per configuration.
    - seed: Random seed for reproducibility.
    """
    plt.figure(figsize=(15, 10))

    for idx, pshare in enumerate(pshare_values):
        print(f"Working on plot for pshare {pshare}...")

        vanilla_errors = {n: [] for n in sample_sizes}
        friendship_errors = {n: [] for n in sample_sizes}
        hybrid_errors = {n: [] for n in sample_sizes}
        fotouhi_errors = {n: [] for n in sample_sizes}

        # Initialize the graph model once for each Pshare
        graph_model = graph_model_class(init_num_nodes=init_num_nodes, alpha=alpha, p_share=pshare, seed=seed)

        # Perform Monte Carlo iterations
        for _ in range(monte_carlo_iterations):
            for n_samples in sample_sizes:
                # Vanilla estimator
                vanilla_estimate = vanilla_estimator(graph_model, n_samples)
                vanilla_error = abs(vanilla_estimate - graph_model.true_average_exposure) / graph_model.true_average_exposure * 100
                vanilla_errors[n_samples].append(vanilla_error)

                # Friendship-paradox-based estimator
                friendship_estimate = friendship_paradox_estimator(graph_model, n_samples)
                friendship_error = abs(friendship_estimate - graph_model.true_average_exposure) / graph_model.true_average_exposure * 100
                friendship_errors[n_samples].append(friendship_error)

                # Hybrid estimator
                hybrid_estimate = hybrid_estimator(graph_model, n_samples)
                hybrid_error = abs(hybrid_estimate - graph_model.true_average_exposure) / graph_model.true_average_exposure * 100
                hybrid_errors[n_samples].append(hybrid_error)

                # fotouhi estimator
                fotouhi_estimate = fotouhi_estimator(graph_model, n_samples)
                fotouhi_error = abs(fotouhi_estimate - graph_model.true_average_exposure) / graph_model.true_average_exposure * 100
                fotouhi_errors[n_samples].append(fotouhi_error)

        # Average the errors over all Monte Carlo iterations
        vanilla_avg_errors = [np.mean(vanilla_errors[n]) for n in sample_sizes]
        friendship_avg_errors = [np.mean(friendship_errors[n]) for n in sample_sizes]
        hybrid_avg_errors = [np.mean(hybrid_errors[n]) for n in sample_sizes]
        fotouhi_avg_errors = [np.mean(fotouhi_errors[n]) for n in sample_sizes]

        print(f"vanilla_avg_errors = {vanilla_avg_errors}")
        print(f"friendship_avg_errors = {friendship_avg_errors}")
        print(f"hybrid_avg_errors = {hybrid_avg_errors}")
        print(f"fotouhi_avg_errors = {fotouhi_avg_errors}")

        # Determine dynamic y-limits for the plot
        all_errors = (
            vanilla_avg_errors + friendship_avg_errors + hybrid_avg_errors + fotouhi_avg_errors
        )
        y_min, y_max = min(all_errors) * 0.9, max(all_errors) * 1.1  # Add some padding

        # Plot results for the current Pshare
        plt.subplot(2, len(pshare_values) // 2, idx + 1)
        plt.plot(sample_sizes, vanilla_avg_errors, label=f"$\\hat{{f}}_{{V}}, p_s = {pshare}$", marker="o", color="red")
        plt.plot(sample_sizes, friendship_avg_errors, label=f"$\\hat{{f}}_{{FP}}, p_s = {pshare}$", marker="s", color="green")
        plt.plot(sample_sizes, hybrid_avg_errors, label=f"$\\hat{{f}}_{{H}}, p_s = {pshare}$", marker="^", color="blue")
        plt.plot(sample_sizes, fotouhi_avg_errors, label=f"$\\hat{{f}}_{{F}}, p_s = {pshare}$", marker="*", color="purple")
        plt.xlabel("Sample Size (n)")
        plt.ylabel("Absolute Error (%)")
        plt.title(f"Absolute Error for $p_s = {pshare}$")
        # plt.ylim(0, 200)  
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.savefig("estimator_performance_strong.png")

if __name__ == "__main__":
    # Run the test with GraphModel class
    # test_estimators_weak(GraphModel, init_num_nodes=10000, alpha=2.5, 
    #                      p_share=0.5, seed=42, sample_sizes=[10, 20, 30, 40, 50])
    

    test_estimators_strong(GraphModel, init_num_nodes=10000, alpha=2.2, 
                           pshare_values=[0.015, 0.025, 0.035, 0.05], 
                           sample_sizes=[10, 20, 30, 40, 50], 
                           monte_carlo_iterations=500, seed=42)

