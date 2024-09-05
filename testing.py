import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Initialize arrays to store lines from files
original_best_obj = []
pruned_best_obj = []
original_total_time = []
pruned_total_time = []


def calculate_statistics(array, name):
    mean = np.mean(array)
    variance = np.var(array)
    std_dev = np.std(array)
    print(f"{name} - Mean: {mean:.3f}, Variance: {variance:.3f}, Standard Deviation: {std_dev:.3f}")


directories_original = os.listdir("out/")
directories_pruned = os.listdir("prunedArcs/out")

# Process directories with 100 files each
for i in range(len(directories_original)):
    original_total_time = []
    pruned_total_time = []
    original_best_obj = []
    pruned_best_obj = []
    directory = directories_original[i]
    # Get all txt files in the directory
    print(directory)
    files = os.listdir("out/" + directory)
    for file in files:
        if file.endswith('.out'):
            file_path = os.path.join("out/" + directory, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()  # Read and strip the lines
                objective_value_line = lines[len(lines) - 5]
                total_run_time = lines[len(lines) - 1]
                original_best_obj.append(float(objective_value_line[11:len(objective_value_line) - 1]))
                original_total_time.append(float(total_run_time[18:len(total_run_time) - 1]))

    original_best_obj = original_best_obj[0:20]
    original_total_time = original_total_time[0:20]

    directory = directories_pruned[i]
    files = os.listdir("prunedArcs/out/" + directory)
    for file in files:
        if file.endswith('.out'):
            file_path = os.path.join("prunedArcs/out/" + directory, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()  # Read and strip the lines
                objective_value_line = lines[len(lines) - 5]
                total_run_time = lines[len(lines) - 1]
                pruned_best_obj.append(float(objective_value_line[11:len(objective_value_line) - 1]))
                pruned_total_time.append(float(total_run_time[18:len(total_run_time) - 1]))

    # Calculate statistics
    calculate_statistics(original_best_obj, "Original Best Obj")
    calculate_statistics(pruned_best_obj, "Pruned Best Obj")
    calculate_statistics(original_total_time, "Original Total Time")
    calculate_statistics(pruned_total_time, "Pruned Total Time")

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
    sns.histplot(original_total_time, color='green', kde=True, label='Original Total Time')
    sns.histplot(pruned_total_time, color='purple', kde=True, label='Pruned Total Time')
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
    plt.boxplot([original_total_time, pruned_total_time], tick_labels=['Original Total Time', 'Pruned Total Time'],
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

    # Print correlation coefficients and linear fit details
    print(
        f"Linear fit for (Original Total Time, Original Best Obj): y = {intercept1:.2f} + {slope1:.2f}x, Correlation coefficient: {r_value1:.3f}")
    print(
        f"Linear fit for (Pruned Total Time, Pruned Best Obj): y = {intercept2:.2f} + {slope2:.2f}x, Correlation coefficient: {r_value2:.3f}")