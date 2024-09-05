import math
import random

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from itertools import product

import util
from sklearn.svm import SVR

vertices = np.array([])
startingPoints = np.array([])
destinationPoints = np.array([])
chargingStationPoints = np.array([])
timeWindows = np.array([])
startingPointsCenter = np.array([])
destinationPointsCenter = np.array([])
distances = np.array([])
neighborhoods = np.array([])
infeasible_arcs = np.array([])


def is_connected(graph):
    """Check if the graph is connected using DFS."""
    n = len(graph)
    visited = [False] * n

    def dfs(v):
        stack = [v]
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                stack.extend(
                    neighbor for neighbor, connected in enumerate(graph[node]) if connected and not visited[neighbor])

    # Start from the first vertex
    dfs(0)

    # Check if all vertices are visited
    return all(visited)


def feature_extractor(i, j):
    dataPoint = []
    # Edge length
    #edge_length = np.linalg.norm(np.array(vertices[i]) - np.array(vertices[j]))
    edge_length = distances[i, j]
    dataPoint.append(edge_length if edge_length != 0 else 100)
    # TODO Do this once as well: Try to calculate an array of lists each containing the sorted neighbourhood for the
    #  given node. Same for nearness. Do it once.
    #dataPoint.append(np.where(neighborhoods[i, :] == vertices[j])[0][0])

    # Distance from start and end points
    #edgeCenter = util.calculate_centroid([vertices[i], vertices[j]])
    #minimumStartingPoint = util.findSmallestDistance(startingPoints, edgeCenter)
    #maximumStartingPoint = util.findBiggestDistance(startingPoints, edgeCenter)
    #minimumDestinationPoint = util.findSmallestDistance(destinationPoints, edgeCenter)
    #maximumDestinationPoint = util.findBiggestDistance(destinationPoints, edgeCenter)
    #dataPoint.append(math.dist(startingPointsCenter, edgeCenter))
    ##dataPoint.append(math.dist(max(startingPoints, key=lambda item: abs(item[0] + item[1])), edgeCenter))
    #dataPoint.append(math.dist(min(startingPoints, key=lambda item: abs(item[0] + item[1])), edgeCenter))
    #dataPoint.append(math.dist(minimumStartingPoint, edgeCenter))
    #dataPoint.append(math.dist(maximumStartingPoint, edgeCenter))

    # distance from end
    # same principle as for start
    #dataPoint.append(math.dist(destinationPointsCenter, edgeCenter))
    #dataPoint.append(math.dist(min(destinationPoints, key=lambda item: abs(item[0] + item[1])), edgeCenter))
    #dataPoint.append(math.dist(max(destinationPoints, key=lambda item: abs(item[0] + item[1])), edgeCenter))
    #dataPoint.append(math.dist(minimumDestinationPoint, edgeCenter))
    #dataPoint.append(math.dist(maximumDestinationPoint, edgeCenter))
    # Clusterness (not that much of a positive effect but omitted due to performance reasons)
    # Charging station edge
    #dataPoint.append(vertices[i] in chargingStationPoints)
    #dataPoint.append(vertices[j] in chargingStationPoints)
    #dataPoint.append(vertices[i] in chargingStationPoints and vertices[j] in chargingStationPoints)

    # Time windows
    tw_i = timeWindows[i]
    tw_j = timeWindows[j]
    dataPoint.append(((tw_i[1] - tw_j[1]) + (tw_i[0] - tw_j[0])) / 2)
    #dataPoint.append(tw_i[1] - tw_j[1])
    #dataPoint.append(tw_i[0] - tw_j[0])
    #   dataPoint.append(tw_i[1] - tw_j[0])

    return dataPoint


"""

def feature_extractor(v):
    i = int(v[0])
    j = int(v[1])
    dataPoint = []
    # Edge length
    #edge_length = np.linalg.norm(np.array(vertices[i]) - np.array(vertices[j]))
    edge_length = distances[i, j]
    dataPoint.append(edge_length if edge_length != 0 else 100)
    # TODO Do this once as well: Try to calculate an array of lists each containing the sorted neighbourhood for the
    #  given node. Same for nearness. Do it once.
    dataPoint.append(np.where(neighborhoods[i, :] == vertices[j])[0][0])


    # Distance from start and end points
    #edgeCenter = util.calculate_centroid([vertices[i], vertices[j]])
    #minimumStartingPoint = util.findSmallestDistance(startingPoints, edgeCenter)
    #maximumStartingPoint = util.findBiggestDistance(startingPoints, edgeCenter)
    #minimumDestinationPoint = util.findSmallestDistance(destinationPoints, edgeCenter)
    #maximumDestinationPoint = util.findBiggestDistance(destinationPoints, edgeCenter)
    #dataPoint.append(math.dist(startingPointsCenter, edgeCenter))
    ##dataPoint.append(math.dist(max(startingPoints, key=lambda item: abs(item[0] + item[1])), edgeCenter))
    #dataPoint.append(math.dist(min(startingPoints, key=lambda item: abs(item[0] + item[1])), edgeCenter))
    #dataPoint.append(math.dist(minimumStartingPoint, edgeCenter))
    #dataPoint.append(math.dist(maximumStartingPoint, edgeCenter))

    # distance from end
    # same principle as for start
    #dataPoint.append(math.dist(destinationPointsCenter, edgeCenter))
    #dataPoint.append(math.dist(min(destinationPoints, key=lambda item: abs(item[0] + item[1])), edgeCenter))
    #dataPoint.append(math.dist(max(destinationPoints, key=lambda item: abs(item[0] + item[1])), edgeCenter))
    #dataPoint.append(math.dist(minimumDestinationPoint, edgeCenter))
    #dataPoint.append(math.dist(maximumDestinationPoint, edgeCenter))
    # Clusterness (not that much of a positive effect but omitted due to performance reasons)
    # Charging station edge
    #dataPoint.append(vertices[i] in chargingStationPoints)
    #dataPoint.append(vertices[j] in chargingStationPoints)
    #dataPoint.append(vertices[i] in chargingStationPoints and vertices[j] in chargingStationPoints)

    # Time windows
    tw_i = timeWindows[i]
    tw_j = timeWindows[j]
    dataPoint.append(((tw_i[1] - tw_j[1]) + (tw_i[0] - tw_j[0])) / 2)
    dataPoint.append(tw_i[1] - tw_j[1])
    dataPoint.append(tw_i[0] - tw_j[0])
    dataPoint.append(tw_i[1] - tw_j[0])

    return dataPoint
"""


def process_directory_and_predict_svr(svr_model, directory_path, l_percent):
    """
    Processes each file in the directory, extracts features, and calls predict_top_l_percent.

    Parameters:
    - svr_model: Trained SVR model
    - directory_path: Path to the directory containing files
    - l_percent: Percentage of top entries to mark as 1 (0 < l_percent <= 100)
    """
    files = os.listdir(directory_path)
    print(files)
    for filename in files:
        print(filename)
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            print(f"Processing file: {filename}")
            with open(file_path, 'r') as file:
                contents = file.readlines()
                nodes = contents[1:len(contents) - 13]

            # Assuming util functions for extracting necessary data
            vertices = util.extract_vertices(nodes)
            timeWindows = util.extract_timeWindow(filename)
            startingPoints = util.extract_startingPoints(contents)
            destinationPoints = util.extract_endingPoints(contents)
            chargingStationPoints = util.extract_ChargingStations(contents)
            infeasible_arcs = util.extract_infeasible_arcs(filename)

            output_array = predict_top_percent_svr(svr_model, l_percent,
                                                   vertices, startingPoints, destinationPoints, chargingStationPoints,
                                                   timeWindows, infeasible_arcs)
            """
            output_array = predict_random(l_percent, vertices, startingPoints, destinationPoints, chargingStationPoints,
                                          timeWindows)
            """
            row_sums = np.sum(output_array, axis=1)
            print(row_sums)
            plt.figure(figsize=(8, 5))
            plt.plot(row_sums, marker='o', linestyle='-', color='b')
            plt.title('Degree for each node ' + filename[:-4])
            plt.xlabel('Node index')
            plt.ylabel('Degree')
            plt.grid(True)
            plt.savefig('degree_svr_' + filename[:-4] + '.png')
            plt.show()

            output_file_path = os.path.join("prunedArcs/" + "svr/" + str(l_percent) + "/", f"{filename}")
            if not is_connected(output_array):
                output_array = "failed"
            output_file_path = os.path.join("prunedArcs/", f"{filename}")

            # Save the output_array to the text file
            with open(output_file_path, 'w') as output_file:
                np.savetxt(output_file, output_array, fmt='%d')


def predict_top_percent_svr(svr_model, percent, v, st, des,
                            ch, time, infeas):
    """
    Predicts values for an n x n array using an SVR model and marks the top l% as 1 and the rest as 0.

    Parameters:
    - svr_model: Trained SVR model
    - input_array: n x n numpy array for which to make predictions
    - feature_extractor: Function that takes an entry (i, j) and returns a feature vector
    - l_percent: Percentage of top entries to mark as 1 (0 < l_percent <= 100)

    Returns:
    - output_array: n x n numpy array with 1s and 0s based on top l% predictions
    """
    n = len(v)
    predictions = np.zeros((n, n))
    global vertices
    global startingPoints
    global destinationPoints
    global chargingStationPoints
    global timeWindows
    global startingPointsCenter
    global destinationPointsCenter
    global distances
    global neighborhoods
    global infeasible_arcs
    vertices = v
    startingPoints = st
    destinationPoints = des
    chargingStationPoints = ch
    timeWindows = time
    startingPointsCenter = util.calculate_centroid(startingPoints)
    destinationPointsCenter = util.calculate_centroid(destinationPoints)
    distances = util.calculateDistanceArray(vertices)
    neighborhoods = util.calculateNeighborhoodArray(np.array(vertices), distances)
    infeasible_arcs = infeas
    features = []
    # First calculate the feature vectors and scale them
    #TODO the loops for chargingStationsPoints etc need a lot of unnecessary time. Dont do n, do reduced length. And set
    # the other variables later. Instead of the range take an array/list with only the indexes. We need the values before
    # so i think we can dodge the double for loop.
    #arr1 = np.arange(7201)
    #arr2 = np.arange(7201)
    #print([[i,j] for i in range(7201) for j in range(7201)])

    #arr = np.array([str(i) + str(j) for i in range(7201) for j in range(7201)])
    #print(arr)
    #vectorized_function = np.vectorize(feature_extractor)
    #print(vectorized_function(arr))
    earliest_time_windows = [t[0] for t in timeWindows]
    print(earliest_time_windows)
    latest_time_windows = [t[1] for t in timeWindows]
    earliest_repeated_time_windows = np.repeat(np.array(earliest_time_windows)[:, None], n, axis=1)

    print(earliest_repeated_time_windows)
    latest_repeated_time_windows = np.repeat(np.array(latest_time_windows)[:, None], n, axis=1)
    average_time_differences = np.add(
        np.subtract(earliest_repeated_time_windows, np.transpose(earliest_repeated_time_windows)),
        np.subtract(latest_repeated_time_windows, np.transpose(latest_repeated_time_windows))) * 1 / 2
    print(average_time_differences)
    features = np.transpose(np.vstack((np.array(distances).flatten(), average_time_differences.flatten())))
    print(features)
    print(features.shape)
    #with open("latest.txt", 'w') as output_file:
    #    np.savetxt(output_file, features, fmt='%d')

    """
    for i in range(n):
        print("Calculated features for one vertice!", i)
        for j in range(n):
            #print("Next Vertice " + str(j))
            if i == j:
                continue
            #elif vertices[i] in chargingStationPoints or vertices[i] in startingPoints or vertices[i] in destinationPoints:
            #    continue
            #elif vertices[j] in chargingStationPoints or vertices[j] in startingPoints or vertices[j] in destinationPoints:
            #    continue
            else:
                feature_vector = feature_extractor(i, j)
                #prediction = svr_model.predict([feature_vector])
                #print(prediction)
                #predictions[i, j] = prediction
                features.append(feature_vector)
    """
    print(infeasible_arcs.shape, np.sum(infeasible_arcs))
    print(np.array(distances).shape)
    features = features[infeasible_arcs == 0]
    print(features)
    print(features.shape)
    print("just after for")
    # scale the featurevector
    scaler = StandardScaler()
    print("Before scaling")
    scaled_features = scaler.fit_transform(features)
    print("scaled the values")
    # predict based on the featurevector
    print(scaled_features)
    standard_edge_predicitions = svr_model.predict(scaled_features)
    print("made the predictions")
    min_val = np.min(standard_edge_predicitions)
    max_val = np.max(standard_edge_predicitions)
    scaled_standard_edge_predicitions = (standard_edge_predicitions - min_val) / (max_val - min_val)
    print("scaled the predictions")
    # Set Charging stations, starting points and destination points edges to 1 and add them
    # to the predictions
    counter = 0
    #zip

    # TODO here the same problem. skip over chargingStationsPoints etc.
    for i in range(n):
        print("Predicted for one vertice!")
        for j in range(n):
            if infeasible_arcs[i * n + j] == 1:
                predictions[i, j] = 0
            else:
                predictions[i, j] = scaled_standard_edge_predicitions[counter]
                counter += 1

        """
        if not (vertices[i] in chargingStationPoints or vertices[i] in startingPoints or vertices[i] in destinationPoints):
            flat_predictions = predictions[i, :].flatten()
            threshold_index = int((1 - percent / 100.0) * len(flat_predictions))
            threshold_value = np.partition(flat_predictions, threshold_index)[threshold_index]
            output_array[i, :] = (predictions[i, :] > threshold_value).astype(int)
        else: output_array[i, :] = predictions[i, :].flatten()
        """
    threshold_index = int((1 - percent / 100.0) * n)
    for i in range(n):
        predictions_single_node = predictions[i, :]
        threshold_value = np.partition(predictions_single_node, threshold_index)[threshold_index]
        pruned_predictions_single_node = (predictions_single_node >= threshold_value).astype(int)
        predictions[i, :] = pruned_predictions_single_node
    """    
    # Flatten the predictions to sort and find the threshold
    flat_predictions = predictions.flatten()
    threshold_index = int((1 - percent / 100.0) * len(flat_predictions))
    threshold_value = np.partition(flat_predictions, threshold_index)[threshold_index]
    # Create the output array based on the threshold
    output_array = (predictions >= threshold_value).astype(int)
    """

    mst = util.calculateMinimumSpanningTree()
    for edge in mst.edges():
        u, v = edge
        predictions[u, v] = 1

    if not is_connected(predictions):
        predictions = "failed"

    print(np.sum(predictions) / (n * n))
    return predictions


def process_directory_and_predict_svm(svm_model, directory_path):
    """
    Processes each file in the directory, extracts features, and calls predict_top_l_percent.

    Parameters:
    - svm_model: Trained SVM model
    - directory_path: Path to the directory containing files
    """
    files = os.listdir(directory_path)

    for filename in files:
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path):
            print(f"Processing file: {filename}")
            with open(file_path, 'r') as file:
                contents = file.readlines()
                nodes = contents[1:len(contents) - 13]

            vertices = util.extract_vertices(nodes)
            timeWindows = util.extract_timeWindow(filename)
            infeasible_arcs = util.extract_infeasible_arcs(filename)
            startingPoints = util.extract_startingPoints(contents)
            destinationPoints = util.extract_endingPoints(contents)
            chargingStationPoints = util.extract_ChargingStations(contents)
            n = len(vertices)

            output_array = predict_top_percent_svm(svm_model,
                                                   vertices, startingPoints, destinationPoints, chargingStationPoints,
                                                   timeWindows, infeasible_arcs)
            print(output_array)
            row_sums = np.sum(output_array, axis=1)

            print(row_sums)
            plt.figure(figsize=(8, 5))
            plt.plot(row_sums, marker='o', linestyle='-', color='b')
            plt.title('Degree for each node ' + filename[:-4])
            plt.xlabel('Node index')
            plt.ylabel('Degree')
            plt.grid(True)
            plt.savefig('degree_svm_' + filename[:-4] + '.png')
            plt.show()

            output_file_path = os.path.join("prunedArcs/", f"{filename}")

            with open(output_file_path, 'w') as output_file:
                np.savetxt(output_file, output_array, fmt='%d')


def predict_top_percent_svm(svm_model, v, st, des, ch, time, infeasible):
    """
    Predicts values for an n x n array using an SVM model and marks the top l% as 1 and the rest as 0.

    Parameters:
    - svm_model: Trained SVM model
    - v: Vertices
    - st: Starting points
    - des: Destination points
    - ch: Charging station points
    - time: Time windows

    Returns:
    - output_array: n x n numpy array with 1s and 0s based on top l% predictions
    """
    n = len(v)
    predictions = np.ones((n, n))

    # Global variables for feature extraction
    global vertices
    global startingPoints
    global destinationPoints
    global chargingStationPoints
    global timeWindows
    global startingPointsCenter
    global destinationPointsCenter
    global distances
    global neighborhoods
    global infeasible_arcs
    vertices = v
    startingPoints = st
    destinationPoints = des
    chargingStationPoints = ch
    timeWindows = time
    startingPointsCenter = util.calculate_centroid(startingPoints)
    destinationPointsCenter = util.calculate_centroid(destinationPoints)
    distances = util.calculateDistanceArray(vertices)
    neighborhoods = util.calculateNeighborhoodArray(np.array(vertices), distances)
    infeasible_arcs = infeasible

    earliest_time_windows = [t[0] for t in timeWindows]
    latest_time_windows = [t[1] for t in timeWindows]
    earliest_repeated_time_windows = np.repeat(np.array(earliest_time_windows)[:, None], n, axis=1)
    latest_repeated_time_windows = np.repeat(np.array(latest_time_windows)[:, None], n, axis=1)
    average_time_differences = np.add(
        np.subtract(earliest_repeated_time_windows, np.transpose(earliest_repeated_time_windows)),
        np.subtract(latest_repeated_time_windows, np.transpose(latest_repeated_time_windows))) * 1 / 2
    features = np.transpose(np.vstack((np.array(distances).flatten(), average_time_differences.flatten())))
    features = features[infeasible_arcs == 0]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print("scaled the values")
    standard_edge_predicitions = svm_model.predict(scaled_features)
    print("made the predictions")
    print("scaled the predictions")
    # Set Charging stations, starting points and destination points edges to 1 and add them
    # to the predictions
    """
    counter = 0
    # Generate predictions
    for i in range(n):
        print("made predicition for one vertice!", i)
        for j in range(n):
            if i == j:
                predictions[i, j] = 0  # No self-loop, no prediction needed
            elif vertices[i] in chargingStationPoints or vertices[i] in startingPoints or vertices[i] in destinationPoints:
                predictions[i, j] = 1
            else:
                feature_vector = feature_extractor(i, j)
                y_proba = svm_model.predict_proba([feature_vector])[:, 1]
                threshold = 0.0003
                prediction = (y_proba >= threshold).astype(int)
                predictions[i, j] = prediction
    """

    # The LNS needs more edges to work. Randomly add more edges
    # To keep run times low this is done by just assigning 1s to
    # a number of random edges. Some of the edges might already be 1
    # but this is faster and the process does not have to be exact.
    # Also important is that all this is done with only the real predictions
    # as the edges with infeasible arcs will be 0 anyways.

    # the counter stores the current position in the predictions. Because only necessary
    # predictions are made this differs from the indices of the resulting predictions array
    # which should hold prediction for every edge including the infeasible ones.

    length_predictions = 0
    length_before = 0

    for i in range(n):
        length_predictions_this_iteration = np.sum(infeasible_arcs[i * n:(i + 1) * n] == 0)
        length_predictions += length_predictions_this_iteration
        predictions_single_node = predictions[i, :]
        random_edges = random.sample(range(0, length_predictions_this_iteration),
                                     math.floor(length_predictions_this_iteration * 0.8))
        standard_edge_predicitions[length_before:length_predictions][random_edges] = 1
        predictions_single_node[infeasible_arcs[i * n:(i + 1) * n] == 0] = standard_edge_predicitions[length_before:
                                                                                                      length_predictions]
        predictions[i, :] = predictions_single_node
        length_before += length_predictions_this_iteration
    print("after assignment of predicitions")
    """
    mst = util.calculateMinimumSpanningTree()
    for edge in mst.edges():
        u, v = edge
        predictions[u, v] = 1

    if not is_connected(predictions):
        predictions = "failed"
    """
    return predictions


def predict_random(percent, v, st, des,
                   ch, time):
    n = len(v)
    global vertices
    global startingPoints
    global destinationPoints
    global chargingStationPoints
    global timeWindows
    global startingPointsCenter
    global destinationPointsCenter
    vertices = v
    startingPoints = st
    destinationPoints = des
    chargingStationPoints = ch
    timeWindows = time
    startingPointsCenter = util.calculate_centroid(startingPoints)
    destinationPointsCenter = util.calculate_centroid(destinationPoints)
    pruned_arcs = util.generate_random_pruned_array(n, percent)
    # First calculate the feature vectors and scale them
    for i in range(n):
        for j in range(n):
            if i == j:
                pruned_arcs[i, j] = 0
            elif vertices[i] in chargingStationPoints or vertices[i] in startingPoints or vertices[
                i] in destinationPoints:
                pruned_arcs[i, j] = 1
            elif vertices[j] in chargingStationPoints or vertices[j] in startingPoints or vertices[
                j] in destinationPoints:
                pruned_arcs[i, j] = 1
    return pruned_arcs
