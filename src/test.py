"""This file contains the test function. It is responsible for brute-testing the simulation, using different parameters."""
import subprocess


def run(args):
    """This runs the testing script, which runs the simulation with different parameters, though the command line.
    Args:
        args (argparse.Namespace): Arguments parsed from the command line
    """

    subprocess.run("chmod +x src/test-script.sh", shell=True)

    grid_sizes = [2, 5, 9]
    queue_capacities = [1, 2, 50]
    congestion_rates = [0.1, 1]
    credit_balances = [1, 20, 50]
    wage_times = [1, 5, 50]

    for grid_size in grid_sizes:
        for queue_capacity in queue_capacities:
            for congestion_rate in congestion_rates:
                for credit_balance in credit_balances:
                    for wage_time in wage_times:
                        print("Running with parameters: --grid_size ", grid_size, " --queue_capacity ", queue_capacity,
                              " --congestion_rate ", congestion_rate, " --credit_balance ", credit_balance, " --wage_time ", wage_time)
                        command = "src/test-script.sh " + str(grid_size) + " " + str(queue_capacity) + " " + str(
                            congestion_rate) + " " + str(credit_balance) + " " + str(wage_time)
                        subprocess.run(command, shell=True)

    print("Test Completed")
