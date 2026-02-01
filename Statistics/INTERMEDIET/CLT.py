# CENTRAL LIMIT THEOREM (CLT) - Comprehensive Python Implementation
# Author: Ashutosh
# Date: February 1, 2026

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Set plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100


def demonstrate_clt_basic():
    """
    Basic demonstration of Central Limit Theorem
    Shows how sample means approach normal distribution regardless of population distribution
    """
    print("=" * 60)
    print("CENTRAL LIMIT THEOREM DEMONSTRATION")
    print("=" * 60)
    
    # Create different population distributions
    np.random.seed(42)
    n_population = 100000
    
    # 1. Exponential distribution (right-skewed)
    exp_pop = np.random.exponential(scale=2.0, size=n_population)
    
    # 2. Uniform distribution
    uniform_pop = np.random.uniform(low=0, high=10, size=n_population)
    
    # 3. Bimodal distribution
    bimodal_pop = np.concatenate([
        np.random.normal(3, 1, n_population//2),
        np.random.normal(8, 1.5, n_population//2)
    ])
    
    populations = [exp_pop, uniform_pop, bimodal_pop]
    pop_names = ['Exponential (Skewed)', 'Uniform', 'Bimodal']
    
    # Population statistics
    print("\nPOPULATION STATISTICS:")
    print("-" * 40)
    for i, (pop, name) in enumerate(zip(populations, pop_names)):
        print(f"{name}:")
        print(f"  Mean: {np.mean(pop):.3f}")
        print(f"  Std:  {np.std(pop):.3f}")
        print(f"  Min:  {np.min(pop):.3f}")
        print(f"  Max:  {np.max(pop):.3f}\n")
    
    # Demonstrate CLT with different sample sizes
    sample_sizes = [5, 30, 100, 500]
    n_samples = 10000
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Central Limit Theorem Demonstration\nDistribution of Sample Means', fontsize=16, fontweight='bold')
    
    for i, (population, pop_name) in enumerate(zip(populations, pop_names)):
        pop_mean = np.mean(population)
        pop_std = np.std(population)
        
        for j, n in enumerate(sample_sizes):
            # Generate sample means
            sample_means = []
            for _ in range(n_samples):
                sample = np.random.choice(population, size=n, replace=True)
                sample_means.append(np.mean(sample))
            
            sample_means = np.array(sample_means)
            theoretical_se = pop_std / np.sqrt(n)
            
            # Plot histogram
            axes[i, j].hist(sample_means, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add theoretical normal curve
            x = np.linspace(sample_means.min(), sample_means.max(), 200)
            y = stats.norm.pdf(x, loc=pop_mean, scale=theoretical_se)
            axes[i, j].plot(x, y, 'r--', linewidth=2, label='Theoretical Normal')
            
            # Formatting
            axes[i, j].set_title(f'{pop_name}\nn={n}, samples={n_samples:,}')
            axes[i, j].set_xlabel('Sample Mean')
            axes[i, j].set_ylabel('Density')
            axes[i, j].axvline(pop_mean, color='green', linestyle='--', linewidth=2, 
                             label=f'Population μ={pop_mean:.2f}')
            axes[i, j].legend(fontsize=8)
            
            # Print statistics
            if i == 0 and j == 0:  # Print only once for brevity
                print(f"SAMPLE MEANS STATISTICS (n={n}):")
                print(f"  Mean of means: {np.mean(sample_means):.4f} (should ≈ {pop_mean:.4f})")
                print(f"  Std of means:  {np.std(sample_means):.4f} (should ≈ {theoretical_se:.4f})")
                print(f"  Theoretical SE: {theoretical_se:.4f}\n")
    
    plt.tight_layout()
    plt.show()


def probability_calculations():
    """
    Demonstrate probability calculations using CLT
    """
    print("=" * 60)
    print("PROBABILITY CALCULATIONS USING CLT")
    print("=" * 60)
    
    # Example 1: Heights of adult males
    print("\nExample 1: Heights of Adult Males")
    print("-" * 40)
    mu = 175  # cm
    sigma = 10  # cm
    n = 25
    
    se = sigma / np.sqrt(n)
    print(f"Population: μ = {mu} cm, σ = {sigma} cm")
    print(f"Sample size: n = {n}")
    print(f"Standard Error: σ/√n = {se:.3f} cm")
    
    # Calculate P(X̄ > 178)
    x_bar = 178
    z_score = (x_bar - mu) / se
    prob = 1 - stats.norm.cdf(z_score)
    
    print(f"P(X̄ > {x_bar} cm) = P(Z > {z_score:.3f}) = {prob:.4f}")
    
    # Example 2: Test scores
    print("\nExample 2: Test Scores")
    print("-" * 40)
    mu = 75
    sigma = 12
    n = 36
    
    se = sigma / np.sqrt(n)
    print(f"Population: μ = {mu}, σ = {sigma}")
    print(f"Sample size: n = {n}")
    print(f"Standard Error: {se:.3f}")
    
    # Calculate P(72 < X̄ < 78)
    x1, x2 = 72, 78
    z1 = (x1 - mu) / se
    z2 = (x2 - mu) / se
    prob = stats.norm.cdf(z2) - stats.norm.cdf(z1)
    
    print(f"P({x1} < X̄ < {x2}) = P({z1:.3f} < Z < {z2:.3f}) = {prob:.4f}")
    
    # Visualization
    x = np.linspace(70, 80, 200)
    y = stats.norm.pdf(x, loc=mu, scale=se)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label=f'Sampling Distribution\nN(μ={mu}, SE={se:.2f})')
    plt.fill_between(x, y, where=(x >= x1) & (x <= x2), alpha=0.3, color='green', 
                     label=f'P({x1} < X̄ < {x2}) = {prob:.4f}')
    plt.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'μ = {mu}')
    plt.axvline(x1, color='green', linestyle=':', linewidth=1)
    plt.axvline(x2, color='green', linestyle=':', linewidth=1)
    plt.xlabel('Sample Mean')
    plt.ylabel('Probability Density')
    plt.title('Sampling Distribution of Sample Means\nUsing Central Limit Theorem')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def sample_size_determination():
    """
    Calculate required sample size for given margin of error
    """
    print("=" * 60)
    print("SAMPLE SIZE DETERMINATION")
    print("=" * 60)
    
    # Example 1: Estimating population mean
    print("\nExample 1: Estimating Average Income")
    print("-" * 40)
    sigma = 15000  # Population standard deviation
    confidence_level = 0.95
    margin_of_error = 1000  # Desired margin of error
    
    # Z-score for 95% confidence
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    
    # Sample size calculation
    n = (z_score * sigma / margin_of_error) ** 2
    n_ceil = int(np.ceil(n))
    
    print(f"Population σ = ${sigma:,}")
    print(f"Confidence level = {confidence_level*100}% (Z = {z_score:.2f})")
    print(f"Margin of error = ±${margin_of_error:,}")
    print(f"Required sample size = {n_ceil:,}")
    
    # Example 2: Different scenarios
    print("\nExample 2: Sample Size for Different Margins of Error")
    print("-" * 50)
    
    scenarios = [
        (500, "±$500"),
        (1000, "±$1,000"),
        (2000, "±$2,000"),
        (5000, "±$5,000")
    ]
    
    results = []
    for margin, label in scenarios:
        n = (z_score * sigma / margin) ** 2
        n_ceil = int(np.ceil(n))
        results.append({
            'Margin of Error': label,
            'Sample Size': f'{n_ceil:,}',
            'n_value': n_ceil
        })
        
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Visualization
    margins = [500, 1000, 2000, 5000]
    sample_sizes = [(z_score * sigma / m) ** 2 for m in margins]
    
    plt.figure(figsize=(10, 6))
    plt.plot(margins, sample_sizes, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Margin of Error ($)')
    plt.ylabel('Required Sample Size')
    plt.title('Sample Size vs Margin of Error\n(95% Confidence, σ = $15,000)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (m, n) in enumerate(zip(margins, sample_sizes)):
        plt.annotate(f'{int(np.ceil(n)):,}', (m, n), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.show()


def clt_for_proportions():
    """
    Demonstrate CLT for sample proportions
    """
    print("=" * 60)
    print("CLT FOR SAMPLE PROPORTIONS")
    print("=" * 60)
    
    # Example: Political poll
    print("\nExample: Political Poll")
    print("-" * 30)
    p = 0.52  # Population proportion
    n = 1000  # Sample size
    
    # Standard error for proportions
    se_p = np.sqrt(p * (1 - p) / n)
    
    print(f"Population proportion: p = {p}")
    print(f"Sample size: n = {n:,}")
    print(f"Standard Error: √[p(1-p)/n] = {se_p:.4f}")
    
    # Calculate probability that sample proportion is within margin
    margin = 0.03  # ±3%
    p1, p2 = p - margin, p + margin
    
    z1 = (p1 - p) / se_p
    z2 = (p2 - p) / se_p
    prob = stats.norm.cdf(z2) - stats.norm.cdf(z1)
    
    print(f"P({p1:.3f} < p̂ < {p2:.3f}) = {prob:.4f}")
    
    # Visualization
    p_hats = np.linspace(0.45, 0.59, 200)
    y = stats.norm.pdf(p_hats, loc=p, scale=se_p)
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_hats, y, 'b-', linewidth=2, 
             label=f'Sampling Distribution of p̂\nN(p={p}, SE={se_p:.4f})')
    plt.fill_between(p_hats, y, where=(p_hats >= p1) & (p_hats <= p2), 
                     alpha=0.3, color='green', 
                     label=f'P({p1:.3f} < p̂ < {p2:.3f}) = {prob:.4f}')
    plt.axvline(p, color='red', linestyle='--', linewidth=2, label=f'p = {p}')
    plt.axvline(p1, color='green', linestyle=':', linewidth=1)
    plt.axvline(p2, color='green', linestyle=':', linewidth=1)
    plt.xlabel('Sample Proportion (p̂)')
    plt.ylabel('Probability Density')
    plt.title('Sampling Distribution of Sample Proportions\nUsing CLT for Proportions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def clt_conditions_check():
    """
    Check conditions for applying CLT
    """
    print("=" * 60)
    print("CLT CONDITIONS CHECK")
    print("=" * 60)
    
    conditions = [
        ("Random Sample", "✓ Data must be collected randomly", True),
        ("Independent Observations", "✓ Each observation independent of others", True),
        ("Finite Population Variance", "✓ Population has finite variance", True),
        ("Sample Size", "✓ n ≥ 30 (rule of thumb)", False),
        ("For Proportions", "✓ np ≥ 5 and n(1-p) ≥ 5", False)
    ]
    
    print("\nCONDITIONS FOR APPLYING CLT:")
    print("-" * 50)
    for condition, description, met in conditions:
        status = "✓ MET" if met else "⚠ CHECK"
        print(f"{status} {condition}: {description}")
    
    print("\nSAMPLE SIZE GUIDELINES:")
    print("-" * 30)
    guidelines = [
        ("Normal population", "Any n is fine", "n ≥ 1"),
        ("Mildly skewed", "Moderate n sufficient", "n ≥ 30"),
        ("Moderately skewed", "Larger n needed", "n ≥ 50"),
        ("Highly skewed", "Very large n required", "n ≥ 100")
    ]
    
    guideline_df = pd.DataFrame(guidelines, 
                               columns=['Population Shape', 'Sample Size', 'Recommendation'])
    print(guideline_df.to_string(index=False))


def main():
    """
    Main function to run all CLT demonstrations
    """
    print("CENTRAL LIMIT THEOREM COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    print("This program demonstrates the Central Limit Theorem through")
    print("visualizations, calculations, and practical examples.\n")
    
    try:
        # Run all demonstrations
        demonstrate_clt_basic()
        probability_calculations()
        sample_size_determination()
        clt_for_proportions()
        clt_conditions_check()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✓ The Central Limit Theorem states that the distribution of sample means")
        print("  approaches a normal distribution as sample size increases, regardless")
        print("  of the population distribution shape.")
        print("\n✓ Key Formula: X̄ ~ N(μ, σ²/n) for large n")
        print("\n✓ Standard Error: SE = σ/√n")
        print("\n✓ Allows statistical inference even when population isn't normal")
        print("\n✓ Sample size rule of thumb: n ≥ 30 (larger for skewed data)")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the required packages installed:")
        print("pip install numpy matplotlib seaborn scipy pandas")

if __name__ == "__main__":
    main()
