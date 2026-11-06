import random
import math
import csv
import statistics
from scipy import stats

SIMULATION_TIME = 500.0

I1 = 1.5
I2 = 4.0

P1_MEAN = 2.0
P1_STD = 0.3

P2_MEAN = 2.5
P2_STD = 0.2

Tm = 0.0

S = 0

n = 0

Q = []

J1 = None
J2 = None

St = 501.0

idle_time = 0.0
queue_area = 0.0

def gen_exp(mean):
    u = random.random()
    return -mean * math.log(1 - u)

def gen_normal(mean, std):
    u1 = 1.0 - random.random()
    u2 = random.random()

    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    return mean + std * z


def run_simulation(seed=None, save_log=False):

    if seed is not None:
        random.seed(seed)

    Tm = 0.0
    S = 0
    n = 0
    Q = []

    J1 = Tm + gen_exp(I1)
    J2 = Tm + gen_exp(I2)
    St = 501.0

    idle_time = 0.0
    queue_area = 0.0

    events = []

    def record_event(event_name):
        if save_log:
            events.append({
                "Event": event_name,
                "Tm": round(Tm, 4),
                "J1": round(J1, 4),
                "J2": round(J2, 4),
                "St": round(St, 4),
                "S": S,
                "n": n,
                "Q": str(Q.copy())
            })

    record_event("Start")

    while True:

        next_time = min(J1, J2, St)

        if next_time > SIMULATION_TIME:

            dt = SIMULATION_TIME - Tm

            idle_time += (1 - S) * dt
            queue_area += n * dt

            Tm = SIMULATION_TIME
            break

        dt = next_time - Tm

        idle_time += (1 - S) * dt
        queue_area += n * dt

        Tm = next_time

        if next_time == J1:
            event_type = "J1"
        elif next_time == J2:
            event_type = "J2"
        else:
            event_type = "E"

        if event_type == "J1" or event_type == "J2":

            if event_type == "J1":
                J1 = Tm + gen_exp(I1)
            else:
                J2 = Tm + gen_exp(I2)

            if S == 0:
                S = 1

                if event_type == "J1":
                    service_time = gen_normal(P1_MEAN, P1_STD)
                else:
                    service_time = gen_normal(P2_MEAN, P2_STD)

                while service_time <= 0:
                    if event_type == "J1":
                        service_time = gen_normal(P1_MEAN, P1_STD)
                    else:
                        service_time = gen_normal(P2_MEAN, P2_STD)

                St = Tm + service_time
            else:
                Q.append(event_type)
                n = len(Q)
        else:

            if len(Q) == 0:
                S = 0
                St = 501.0

            else:
                job_type = Q.pop(0)
                n = len(Q)

                if job_type == "J1":
                    service_time = gen_normal(P1_MEAN, P1_STD)
                else:
                    service_time = gen_normal(P2_MEAN, P2_STD)

                while service_time <= 0:
                    if job_type == "J1":
                        service_time = gen_normal(P1_MEAN, P1_STD)
                    else:
                        service_time = gen_normal(P2_MEAN, P2_STD)

            St = Tm + service_time
        
        record_event(event_type)

    record_event("End")

    MoE1 = idle_time / SIMULATION_TIME
    MoE2 = queue_area / SIMULATION_TIME

    if save_log:

        fieldnames = [
            "Event", "Tm", "J1", "J2",
            "St", "S", "n", "Q"
        ]

        with open("simulation_events.csv", "w",
                  newline="", encoding="utf-8") as file:

            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)

        with open("simulation_events_summary.csv", "w",
                  newline="", encoding="utf-8") as file:

            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events[:20])
            writer.writerows(events[-20:])

        print(f"Event log saved: {len(events)} records")
        print("First and last 20 events saved.")

    return MoE1, MoE2


def calculate_statistics(values, name):
    n = len(values)

    mean = statistics.mean(values)
    median = statistics.median(values)
    std = statistics.stdev(values)
    variance = statistics.variance(values)

    standard_error = std / math.sqrt(n)
    t_critical = stats.t.ppf(0.975, df=n - 1)

    margin = t_critical * standard_error

    ci_lower = mean - margin
    ci_upper = mean + margin

    modes = statistics.multimode(values)

    if len(modes) == n:
        mode_result = "No unique mode"
    else:
        mode_result = str(modes)

    print(f"\n=== STATISTICS FOR {name} ===")
    print(f"Mean: {mean:.6f}")
    print(f"Median: {median:.6f}")
    print(f"Mode: {mode_result}")
    print(f"Standard deviation: {std:.6f}")
    print(f"Variance: {variance:.6f}")
    print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")


if __name__ == "__main__":

    results = []

    print("=== TEN SIMULATION RUNS ===")

    for run in range(1, 11):

        seed = 85121 + run - 1

        save_log = (run == 1)

        MoE1, MoE2 = run_simulation(
            seed=seed,
            save_log=save_log
        )

        results.append({
            "Run": run,
            "Seed": seed,
            "MoE1": MoE1,
            "MoE2": MoE2
        })

        print(
            f"Run {run:2d}: "
            f"MoE1 = {MoE1:.6f}, "
            f"MoE2 = {MoE2:.4f}"
        )

    with open("simulation_10_runs.csv", "w",
              newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["Run", "Seed", "MoE1", "MoE2"]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to simulation_10_runs.csv")

    MoE1_values = [row["MoE1"] for row in results]
    MoE2_values = [row["MoE2"] for row in results]

    calculate_statistics(
        MoE1_values,
        "MoE1 - Downtime Factor"
    )

    calculate_statistics(
        MoE2_values,
        "MoE2 - Average Queue Length"
    )

