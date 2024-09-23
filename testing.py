import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from scipy import stats

########################################################################################################################
##
##                    -- Testing --
## This file is for testing and comparing the results of the pruned and original
## instances of the LNS .
##
########################################################################################################################startingPoints = np.array([])




# Initialize arrays to store lines from files
original_best_obj = []
pruned_best_obj = []
original_total_time = []
pruned_total_time = []
invalid_options = ["a180-3600_26.out", "a200-4000_35.out", "a220-4400_05.out",
                   "a240-4800_10.out", "a260-5200_25.out", "a260-5200_04.out"]


def calculate_statistics(array, name):
    """
    Calculates mean, median, variance and standard deviation for the given list and
    also returns it as a dictionary with the namen given
    :param array: list to calculate statistics
    :param name: name to add to the calculated statistics
    :return: the calculated statistics
    """
    mean = np.mean(array)
    median = np.median(array)
    variance = np.var(array)
    std_dev = np.std(array)
    return {
        "name": name,
        "mean": mean,
        "median": median,
        "variance": variance,
        "std_dev": std_dev
    }


def percentage_difference(original_value, pruned_value):
    """
    Calculated the percentage difference between the two given values.
    :param original_value: first value to compare
    :param pruned_value: second value to compare
    :return: the percentage difference
    """
    if original_value != 0:
        return ((original_value - pruned_value) / original_value) * 100
    return 0  # Avoid division by zero


def compare_statistics(original, pruned, original_name, pruned_name, instance_name):
    """
    This takes 2 lists, calculates statistics for each of them, compares those statistics
    both in absolut and percentage terms and prints the results with the given names.

    :param original: first list to calculate statistics for
    :param pruned: second list to calculate statistics for
    :param original_name: the name that should be set for the statistic results of the first list original
    :param pruned_name: the name that should be set for the statistic results of the second list pruned
    :param instance_name: the main title of the printed table
    :return: nothing is returned. The results are only printed.
    """
    original_stats = calculate_statistics(original, original_name)
    pruned_stats = calculate_statistics(pruned, pruned_name)

    # Print comparison results
    print(f"Comparison between {original_name} and {pruned_name} for {instance_name}:")
    print(f"{'Statistic':<20} {'Original':<15} {'Pruned':<15} {'Difference':<15} {'% Difference':<15}")

    for stat in ['mean', 'median', 'variance', 'std_dev']:
        original_value = original_stats[stat]
        pruned_value = pruned_stats[stat]
        difference = original_value - pruned_value
        percent_diff = percentage_difference(original_value, pruned_value)

        print(f"{stat.capitalize():<20} {original_value:<15.3f} {pruned_value:<15.3f}"
              f" {difference:<15.3f} {percent_diff:<10.2f}%")


directories_original = os.listdir("out/l2")
directories_pruned = os.listdir("prunedArcs/out/l2")

# Process directories with 100 files each
for i in range(len(directories_original)):
    original_total_time = []
    pruned_total_time = []
    original_best_obj = []
    pruned_best_obj = []
    directory = directories_original[i]
    # Get all txt files in the directory
    files = os.listdir("out/l2/" + directory)
    for file in files:
        if file in ["a180-3600_26.out", "a200-4000_35.out", "a220-4400_05.out", "a260-5200_25.out"]:
            continue
        if file.endswith('.out'):
            file_path = os.path.join("out/l2/" + directory, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()  # Read and strip the lines
                objective_value_line = lines[len(lines) - 5]
                total_run_time = lines[len(lines) - 1]
                #try:
                value = float(objective_value_line[11:len(objective_value_line) - 1])
                original_best_obj.append(value)
                original_total_time.append(float(total_run_time[18:len(total_run_time) - 1]))
                #except ValueError:
                #    print("Invalid original file: " + file)
                #    invalid_options.append(file)

    directory = directories_pruned[i]
    files = os.listdir("prunedArcs/out/l2/" + directory)
    for file in files:
        if file in ["a240-4800_10.out", "a260-5200_04.out"]:
            continue
        if file.endswith('.out'):
            file_path = os.path.join("prunedArcs/out/l2/" + directory, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()  # Read and strip the lines
                objective_value_line = lines[len(lines) - 5]
                total_run_time = lines[len(lines) - 1]
                #try:
                value = float(objective_value_line[11:len(objective_value_line) - 1])
                pruned_best_obj.append(value)
                pruned_total_time.append(float(total_run_time[18:len(total_run_time) - 1]))
                #except ValueError:
                #    print("Invalid pruned instance file: " + file)
                #    invalid_options.append(file)

    # Calculate statistics
    compare_statistics(original_best_obj, pruned_best_obj, "Original Best Obj", "Pruned Best Obj", directory)
    compare_statistics(original_total_time, pruned_total_time, "Original Total Time", "Pruned Total Time", directory)
    """
    boxcox_original_total_time, lambda_ = stats.boxcox(np.array(original_total_time))
    print(f"Optimal λ: {lambda_}")
    boxcox_pruned_total_time, lambda_ = stats.boxcox(np.array(pruned_total_time))
    print(f"Optimal λ: {lambda_}")
    """
    log_original_total_time = np.log(np.array(original_total_time))/np.log(10)
    log_pruned_total_time = np.log(np.array(pruned_total_time))/np.log(10)

    # Visualization for individual distributions

    plt.figure(figsize=(14, 7))

    # Histogram for objective values
    plt.subplot(2, 2, 1)
    sns.histplot(original_best_obj, color='blue', kde=True, label='Original Best Obj')
    sns.histplot(pruned_best_obj, color='red', kde=True, label='Pruned Best Obj')
    plt.legend()
    plt.title('Distribution of Original Best Obj and Pruned Best Obj')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Histogram for total time
    plt.subplot(2, 2, 2)
    sns.histplot(log_original_total_time, color='green', kde=True, label='Original Total Time', bins=50)
    sns.histplot(log_pruned_total_time, color='purple', kde=True, label='Pruned Total Time', bins=50)
    plt.legend()
    plt.title('Distribution of Original Total Time and Pruned Total Time')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Boxplot for objective values
    plt.subplot(2, 2, 3)
    plt.boxplot([original_best_obj, pruned_best_obj], tick_labels=['Original Best Obj', 'Pruned Best Obj'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))
    plt.title('Boxplot Comparison for Original Best Obj and Pruned Best Obj')
    plt.ylabel('Value')

    # Boxplot for total time
    plt.subplot(2, 2, 4)
    plt.boxplot([log_original_total_time, log_pruned_total_time], tick_labels=['Original Total Time', 'Pruned Total Time'],
                patch_artist=True,
                boxprops=dict(facecolor='lightgreen', color='green'),
                medianprops=dict(color='purple'))
    plt.title('Boxplot Comparison for Original Total Time and Pruned Total Time')
    plt.ylabel('Value')

    # Show individual distribution plots
    plt.suptitle('Instance ' + directory, fontsize=16)
    plt.tight_layout()
    plt.savefig("plots/" + directory + ".png")
    plt.show()

    # Combined analysis of (original total time, original best obj) and (pruned total time, pruned best obj)
    plt.figure(figsize=(14, 7))

    # Scatter plot for (original total time, original best obj)
    plt.subplot(1, 2, 1)
    plt.scatter(original_total_time, original_best_obj, color='blue', label='(Original Total Time, Original Best Obj)')
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(original_total_time, original_best_obj)
    plt.plot(original_total_time, intercept1 + slope1 * np.array(original_total_time), 'r',
             label=f'Fit line: y = {intercept1:.2f} + {slope1:.2f}x')
    plt.xlabel('Original Total Time')
    plt.ylabel('Original Best Obj')
    plt.title('Scatter plot and Linear Fit for (Original Total Time, Original Best Obj)')
    plt.legend()
    plt.grid(True)

    # Scatter plot for (pruned total time, pruned best obj)
    plt.subplot(1, 2, 2)
    plt.scatter(pruned_total_time, pruned_best_obj, color='green', label='(Pruned Total Time, Pruned Best Obj)')
    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(pruned_total_time, pruned_best_obj)
    plt.plot(pruned_total_time, intercept2 + slope2 * np.array(pruned_total_time), 'r',
             label=f'Fit line: y = {intercept2:.2f} + {slope2:.2f}x')
    plt.xlabel('Pruned Total Time')
    plt.ylabel('Pruned Best Obj')
    plt.title('Scatter plot and Linear Fit for (Pruned Total Time, Pruned Best Obj)')
    plt.legend()
    plt.grid(True)

    # Show combined scatter plots
    plt.tight_layout()
    plt.show()
    """
    # Print correlation coefficients and linear fit details
    print(
        f"Linear fit for (Original Total Time, Original Best Obj): y = {intercept1:.2f} + {slope1:.2f}x, Correlation coefficient: {r_value1:.3f}")
    print(
        f"Linear fit for (Pruned Total Time, Pruned Best Obj): y = {intercept2:.2f} + {slope2:.2f}x, Correlation coefficient: {r_value2:.3f}")
   """
