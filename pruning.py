import numpy as np
import os
import util
from sklearn.svm import SVR
vertices = np.array([])
startingPoints = np.array([])
destinationPoints = np.array([])
chargingStationPoints = np.array([])
timeWindows = np.array([])
startingPointsCenter = np.array([])
destinationPointsCenter = np.array([])
import math

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
                stack.extend(neighbor for neighbor, connected in enumerate(graph[node]) if connected and not visited[neighbor])

    # Start from the first vertex
    dfs(0)

    # Check if all vertices are visited
    return all(visited)

def feature_extractor(i, j):

    dataPoint = []
    # Edge length
    edge_length = np.linalg.norm(np.array(vertices[i]) - np.array(vertices[j]))
    dataPoint.append(edge_length if edge_length != 0 else 100)

    # Nearness of nodes (simple example, replace with actual logic if different)
    sorted_neighbourhood = sorted(vertices, key=lambda x: np.linalg.norm(np.array(vertices[i]) - np.array(x)))
    nearness = np.linalg.norm(
        np.array(vertices[i]) - np.array(sorted_neighbourhood[1]))  # Example: distance to the closest neighbor
    dataPoint.append(nearness)

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
    # Clusterness
    clusterness = len([v for v in vertices if np.linalg.norm(np.array(v) - np.array(vertices[i])) < 5])
    dataPoint.append(clusterness)
    dataPoint.append(edge_length <= 5)
    # Charging station edge
    #dataPoint.append(vertices[i] in chargingStationPoints)
    #dataPoint.append(vertices[j] in chargingStationPoints)
    #dataPoint.append(vertices[i] in chargingStationPoints and vertices[j] in chargingStationPoints)

    # Time windows
    if vertices[j] in chargingStationPoints or vertices[j] in startingPoints or vertices[j] in destinationPoints:
        dataPoint.extend([0, 0, 0, 0])
    else:
        tw_i = timeWindows[i]
        tw_j = timeWindows[j]
        dataPoint.append(((tw_i[1] - tw_j[1]) + (tw_i[0] - tw_j[0])) / 2)
        dataPoint.append(tw_i[1] - tw_j[1])
        dataPoint.append(tw_i[0] - tw_j[0])
        dataPoint.append(tw_i[1] - tw_j[0])

    return dataPoint


def process_directory_and_predict_svr(svr_model, directory_path, l_percent):
    """
    Processes each file in the directory, extracts features, and calls predict_top_l_percent.

    Parameters:
    - svr_model: Trained SVR model
    - directory_path: Path to the directory containing files
    - l_percent: Percentage of top entries to mark as 1 (0 < l_percent <= 100)
    """
    files = os.listdir(directory_path)

    for filename in files:
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

            # Create an n x n array based on the number of vertices
            n = len(vertices)
            input_array = np.random.rand(n, n)  # Example random array, replace with actual data if available

            # Call the predict_top_l_percent function
            output_array = predict_top_percent_svr(svr_model, l_percent,
                                                 vertices, startingPoints, destinationPoints, chargingStationPoints,
                                                 timeWindows)
            if not is_connected(output_array):
                output_array = "failed"
            output_file_path = os.path.join("prunedArcs/", f"{filename}")

            # Save the output_array to the text file
            with open(output_file_path, 'w') as output_file:
                np.savetxt(output_file, output_array, fmt='%d')

def predict_top_percent_svr(svr_model, percent, v, st, des,
                        ch, time):
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
    vertices = v
    startingPoints = st
    destinationPoints = des
    chargingStationPoints = ch
    timeWindows = time
    startingPointsCenter = util.calculate_centroid(startingPoints)
    destinationPointsCenter = util.calculate_centroid(destinationPoints)
    output_array = np.zeros((n, n))
    # Generate predictions
    for i in range(n):
        for j in range(n):
            if i == j:
                predictions[i, j] = 0
            elif vertices[i] in chargingStationPoints or vertices[i] in startingPoints or vertices[i] in destinationPoints:
                predictions[i, j] = 1
            elif vertices[j] in chargingStationPoints or vertices[j] in startingPoints or vertices[j] in destinationPoints:
                predictions[i, j] = 1
            else:
                feature_vector = feature_extractor(i, j)
                predictions[i, j] = svr_model.predict([feature_vector])
        """
        if not (vertices[i] in chargingStationPoints or vertices[i] in startingPoints or vertices[i] in destinationPoints):
            flat_predictions = predictions[i, :].flatten()
            threshold_index = int((1 - percent / 100.0) * len(flat_predictions))
            threshold_value = np.partition(flat_predictions, threshold_index)[threshold_index]
            output_array[i, :] = (predictions[i, :] > threshold_value).astype(int)
        else: output_array[i, :] = predictions[i, :].flatten()
        """
    # Flatten the predictions to sort and find the threshold
    flat_predictions = predictions.flatten()
    threshold_index = int((1 - percent / 100.0) * len(flat_predictions))
    threshold_value = np.partition(flat_predictions, threshold_index)[threshold_index]
    # Create the output array based on the threshold
    output_array = (predictions >= threshold_value).astype(int)
    mst = util.calculateMinimumSpanningTree()
    for edge in mst.edges():
        u, v = edge
        output_array[u, v] = 1

    if not is_connected(predictions):
        output_array = "failed"

    return output_array


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
            startingPoints = util.extract_startingPoints(contents)
            destinationPoints = util.extract_endingPoints(contents)
            chargingStationPoints = util.extract_ChargingStations(contents)

            # Create an n x n array based on the number of vertices
            n = len(vertices)

            output_array = predict_top_percent_svm(svm_model,
                                                   vertices, startingPoints, destinationPoints, chargingStationPoints,
                                                   timeWindows)
            print(output_array)

            output_file_path = os.path.join("prunedArcs/", f"{filename}")

            # Save the output_array to the text file
            with open(output_file_path, 'w') as output_file:
                np.savetxt(output_file, output_array, fmt='%d')


def predict_top_percent_svm(svm_model, v, st, des, ch, time):
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
    predictions = np.zeros((n, n))

    # Global variables for feature extraction
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

    # Generate predictions
    for i in range(n):
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
    mst = util.calculateMinimumSpanningTree()
    for edge in mst.edges():
        u, v = edge
        predictions[u, v] = 1

    if not is_connected(predictions):
        predictions = "failed"
    return predictions
