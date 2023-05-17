import random

def estimate_pi(num_points):
    points_inside_circle = 0
    total_points = num_points

    for _ in range(num_points):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        distance = x**2 + y**2

        # Introduce uncertainty by adding noise to the distance calculation
        distance += random.gauss(0, 0.1)

        if distance <= 1:
            points_inside_circle += 1

    pi_estimate = 4 * (points_inside_circle / total_points)
    return pi_estimate

# Perform Monte Carlo simulation to estimate the value of pi with uncertain methods
//num_simulations = 100000
num_simulations = 100
pi_estimates = []

for _ in range(num_simulations):
    pi_estimate = estimate_pi(1000)
    pi_estimates.append(pi_estimate)

average_pi = sum(pi_estimates) / num_simulations

print(f"Estimated value of pi: {average_pi}")
