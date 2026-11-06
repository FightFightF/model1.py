import random
import math
import csv
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def gen_exp(mean):
    u = random.random()
    return -mean * math.log(1 - u)

def gen_normal(mean, std):
    u1 = 1.0 - random.random()
    u2 = random.random()

    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    return mean + std * z

SAMPLE_SIZE = 1000

I1_samples = [gen_exp(1.5) for _ in range(SAMPLE_SIZE)]
I2_samples = [gen_exp(4.0) for _ in range(SAMPLE_SIZE)]

P1_samples = [gen_normal(2.0, 0.3) for _ in range(SAMPLE_SIZE)]
P2_samples = [gen_normal(2.5, 0.2) for _ in range(SAMPLE_SIZE)]

with open("rng_samples.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(["I1", "I2", "P1", "P2"])

    for i in range(SAMPLE_SIZE):
        writer.writerow([
            I1_samples[i],
            I2_samples[i],
            P1_samples[i],
            P2_samples[i]
        ])

print("Random number generation completed!")
print(f"Sample size: {SAMPLE_SIZE}")
print("Samples saved to rng_samples.csv")

def show_statistics(name, samples, expected_mean, expected_std=None):
    empirical_mean = statistics.mean(samples)
    empirical_std = statistics.stdev(samples)

    print(f"\nDistribution: {name}")
    print(f"Theoretical mean: {expected_mean:.4f}")
    print(f"Empirical mean:   {empirical_mean:.4f}")

    if expected_std is not None:
        print(f"Theoretical standard deviation: {expected_std:.4f}")

    print(f"Empirical standard deviation:   {empirical_std:.4f}")


print("\n=== RANDOM NUMBER GENERATOR STATISTICS ===")

show_statistics("I1 - Exponential(1.5)", I1_samples, 1.5, 1.5)
show_statistics("I2 - Exponential(4.0)", I2_samples, 4.0, 4.0)

show_statistics("P1 - Normal(2, 0.3)", P1_samples, 2.0, 0.3)
show_statistics("P2 - Normal(2.5, 0.2)", P2_samples, 2.5, 0.2)

print("\n=== EXPONENTIAL RATE COMPARISON ===")

for name, samples, theoretical_mean in [
    ("I1", I1_samples, 1.5),
    ("I2", I2_samples, 4.0)
]:
    theoretical_rate = 1 / theoretical_mean
    empirical_rate = 1 / statistics.mean(samples)

    print(f"\n{name}:")
    print(f"Theoretical rate: {theoretical_rate:.4f}")
    print(f"Empirical rate:   {empirical_rate:.4f}")


def plot_distribution(name, samples, dist_type, mean, std=None):
    plt.figure(figsize=(9, 5))

    plt.hist(
        samples,
        bins=30,
        density=True,
        alpha=0.6,
        label="Empirical distribution"
    )

    if dist_type == "exponential":
        x = np.linspace(0, max(samples), 500)
        y = (1 / mean) * np.exp(-x / mean)

    elif dist_type == "normal":
        x = np.linspace(min(samples), max(samples), 500)
        y = (
            1 / (std * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x - mean) / std) ** 2)
        )

    plt.plot(
        x,
        y,
        linewidth=2,
        label="Theoretical distribution"
    )

    plt.title(name)
    plt.xlabel("Generated value")
    plt.ylabel("Probability density")
    plt.legend()
    plt.grid(alpha=0.25)

    filename = name.split(" - ")[0] + "_histogram.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


plot_distribution(
    "I1 - Exponential(1.5)",
    I1_samples,
    "exponential",
    1.5
)

plot_distribution(
    "I2 - Exponential(4.0)",
    I2_samples,
    "exponential",
    4.0
)

plot_distribution(
    "P1 - Normal(2, 0.3)",
    P1_samples,
    "normal",
    2.0,
    0.3
)

plot_distribution(
    "P2 - Normal(2.5, 0.2)",
    P2_samples,
    "normal",
    2.5,
    0.2
)

print("\n=== KOLMOGOROV-SMIRNOV TESTS ===")

def perform_ks_test(name, samples, distribution, parameters):

    statistic, p_value = stats.kstest(
        samples,
        distribution,
        args=parameters
    )

    print(f"\nDistribution: {name}")
    print(f"KS statistic: {statistic:.6f}")
    print(f"P-value: {p_value:.6f}")

    if p_value > 0.05:
        print("Result: Fail to reject H0")
    else:
        print("Result: Reject H0")


perform_ks_test(
    "I1 - Exponential(1.5)",
    I1_samples,
    "expon",
    (0, 1.5)
)

perform_ks_test(
    "I2 - Exponential(4.0)",
    I2_samples,
    "expon",
    (0, 4.0)
)

perform_ks_test(
    "P1 - Normal(2, 0.3)",
    P1_samples,
    "norm",
    (2.0, 0.3)
)

perform_ks_test(
    "P2 - Normal(2.5, 0.2)",
    P2_samples,
    "norm",
    (2.5, 0.2)
)

print("\n=== CHI-SQUARE TESTS ===")

def perform_chi_square_test(name, samples, distribution, parameters):

    dist = getattr(stats, distribution)(*parameters)

    probabilities = np.linspace(0, 1, 11)
    boundaries = dist.ppf(probabilities)

    observed, _ = np.histogram(samples, bins=boundaries)

    expected = np.full(10, len(samples) / 10)

    statistic, p_value = stats.chisquare(
        f_obs=observed,
        f_exp=expected
    )

    print(f"\nDistribution: {name}")
    print("Observed frequencies:", observed)
    print("Expected frequencies:", expected.astype(int))
    print(f"Chi-square statistic: {statistic:.6f}")
    print("Degrees of freedom: 9")
    print(f"P-value: {p_value:.6f}")

    if p_value > 0.05:
        print("Result: Fail to reject H0")
    else:
        print("Result: Reject H0")


perform_chi_square_test(
    "I1 - Exponential(1.5)",
    I1_samples,
    "expon",
    (0, 1.5)
)

perform_chi_square_test(
    "I2 - Exponential(4.0)",
    I2_samples,
    "expon",
    (0, 4.0)
)

perform_chi_square_test(
    "P1 - Normal(2, 0.3)",
    P1_samples,
    "norm",
    (2.0, 0.3)
)

perform_chi_square_test(
    "P2 - Normal(2.5, 0.2)",
    P2_samples,
    "norm",
    (2.5, 0.2)
)

