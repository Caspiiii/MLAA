import numpy as np
import math as math
import networkx as nx
import os

########################################################################################################################
##
##                          -- UTIL --
##  General util file for varying extractions, calculations and helpers.
##
########################################################################################################################startingPoints = np.array([])


verticesStored = np.array([])
timeWindowsStored = np.array([])
infeasible_arcs_stored = np.array([])

def calculate_centroid(points):
    """
    For both coordinates this function calculates the average value over all points
    given in points.
    :param points: A list of points
    :return: One point that is the average of all points in the list points
    """
    sum_x = 0
    sum_y = 0
    n = len(points)
    for i in range(len(points)):
        sum_x += points[i][0]
        sum_y += points[i][1]
    centroid_x = sum_x / n
    centroid_y = sum_y / n
    return centroid_x, centroid_y

def extract_startingPoints(instance):
    """
    Extracts the origin depots of the given instance. This does not read in an instance
    but only extracts the origin depots (coordinates). Also, verticesStored has to be set.
    :param instance: An instance given as a list of strings
    :return: All origin depots of the given instance ( as tuples of coordinates, not indexes).
    """
    startingPointIndex = instance[len(instance) - 13]
    startingPointsIndex = [eval(i) for i in instance[len(instance) - 11].split()]
    startingPointsIndex.append(eval(startingPointIndex))
    startingPoints = []
    for i in range(len(startingPointsIndex)):
        startingPoints.append(verticesStored[int(startingPointsIndex[i])])
    return startingPoints

def extract_first_startingPoints_index(instance):
    """
    Extracts the index of the first origin depot of the given instance. This index represents
    the switch from request locations to other locations.
    :param instance: An instance given as a list of strings
    :return: The index of the first origin depot
    """
    return int(instance[len(instance) - 13])

def extract_endingPoints(instance):
    """
    Extracts the destination depots of the given instance. This does not read in an instance
    but only extracts the destination depots (coordinates). Also, verticesStored has to be set.
    :param instance: An instance given as a list of strings
    :return: All destination depots of the given instance (as tuples of coordinates, not indexes).
    """
    endingPointIndex = instance[len(instance) - 12]
    endingPointsIndex = [eval(i) for i in instance[len(instance) - 10].split()]
    endingPointsIndex.append(eval(endingPointIndex))
    endingPoints = []
    for i in range(len(endingPointsIndex)):
        endingPoints.append(verticesStored[int(endingPointsIndex[i])])
    return endingPoints


def extract_charging_stations(instance):
    """
    Extracts the charging stations of the given instance. This does not read in an instance
    but only extracts the charging stations (coordinates). Also, verticesStored has to be set.
    :param instance: An instance given as a list of strings
    :return: All charging stations of the given instance (as tuples of coordinates, not indexes)
    """
    chargingStationIndex = instance[len(instance) - 9].split()
    chargingStations = []
    for i in range(len(chargingStationIndex)):
        chargingStations.append(verticesStored[int(chargingStationIndex[i])-1])
    return chargingStations

def extract_objectiveFun(solution):
    """
    Extracts the objective function value of the given solution. This is based on the
    .out file, not the .sol file. The file reading has to be done beforehand.
    :param solution: The LNS solution as a list of strings
    :return: The objective function value as a float or -1 if there was an error
    """
    objectiveFunction = find_lines_with_substring(solution, "T best obj:")
    if (len(objectiveFunction) != 1):
        return -1
    return float(objectiveFunction[0].split()[3])

def extract_run_time(solution):
    """
    Extracts the total run time value of the given solution. This is based on the
    .out file, not the .sol file. The file reading has to be done beforehand.
    :param solution: The LNS solution as a list of strings
    :return: The total run time value as a float or -1 if there was an error
    """
    total_run_time = find_lines_with_substring(solution, "T total time")
    if (len(total_run_time) != 1):
        return -1
    return float(total_run_time[0].split()[4])


def extract_vertices(instance):
    """
    Extracts the vertices of the given instance. This does not read in an instance
    but only extracts the vertices (coordinates). Also, verticesStored is set.
    :param instance: An instance given as a list of strings
    :return: All vertices of the given instance (coordinates, not indexes).
    """
    vertices = []
    for each in instance:
        preCoord = each.split()
        xCoord = float(preCoord[1])
        yCoord = float(preCoord[2])
        vertices.append((xCoord, yCoord))
    global verticesStored
    verticesStored = vertices
    return vertices

def extract_timeWindow(instance_path):
    """
    Reads the file at instance_path and extracts the tightened time windows of the given instance. These are the earliest and
    latest service time after the time window tightening in the Preprocessing of the LNS.
    Also, timeWindowsStored is set.
    :param instance_path: The path to the file containing the tightened time windows
    for the given instance.
    :return: The extracted tightened time windows as a list of tuples.
    """
    timeWindow = []
    file_path_time_windows = os.path.join(instance_path)
    if os.path.isfile(file_path_time_windows):
        with open(file_path_time_windows, 'r') as file:
            for line in file:
                values = line.split(" ")
                startTime = float(values[0])
                endTime = float(values[1])
                timeWindow.append((startTime, endTime))
    global timeWindowsStored
    timeWindowsStored = timeWindow
    return timeWindow

def extract_infeasible_arcs(instance_path):
    """
    Reads the file at instance_path and extracts the infeasible arcs based on
    the deterministic preprocessing of the given instance. Also, infeasible_arcs_stored is
    set.
    :param instance_path: The path to the file containing the infeasible arcs for the given instance.
    :return: The infeasible arcs as a nxn numpy array with 1 indicating that the arc between row number and column number
     (which represent vertices as n is the number of vertices) is feasible and 0 if it is infeasible.
    """
    infeasible_arcs = []
    file_path_infeasible_arcs = os.path.join("infeasibleArcs/", instance_path)
    with open(file_path_infeasible_arcs, 'r') as file:
        for line in file:
            values = line.split(" ")
            infeasible_arcs.append([1 if value == "true" else 0 for value in values[0:-1]])
    global infeasible_arcs_stored
    infeasible_arcs_stored = infeasible_arcs
    return np.array(infeasible_arcs).flatten()

def find_lines_with_substring(solution, substring):
    """
    Returns all lines that contain substring.
    :param solution: List of strings
    :param substring: Substring to find in each line
    :return: All lines that contain substring
    """
    result = []
    for line in solution:
        if substring in line:
            result.append(line)
    return result


def extract_route(route):
    """
    Takes a route as a string (as it appears in the .sol files) and extracts the indeces
    of all vertices in the route
    :param route: String representing a route (based on .sol file structure)
    :return: A list of vertice indexes of all vertices that are part of the route
    """
    verticeIndeces = []
    splitted = route.split(", ")
    verticeIndeces.append(int(splitted[0][2:]))
    withoutFirst = splitted[1:len(splitted) - 1]
    for s in withoutFirst:
        if "]" in s:
            verticeIndeces.append(int(s[0:len(s) - 1]))
            break
        verticeIndeces.append(int(s))
    return verticeIndeces


def find_smallest_distance(points, point):
    """
    Finds the point in points that is closest (based on Euclidean distance) to the point given
    in parameter point.
    :param points: List of points with each point being a tuple containing coordinates
    :param point: Point as a tuple containing two coordinates
    :return: Closest point to the given point
    """
    closest_point = None
    smallest_distance = float('inf')  # Initialize with a very large number
    for p in points:
        distance = math.dist(p, point)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_point = p
    return closest_point


def find_biggest_distance(points, point):
    """
    Finds the point in points that is furthest (based on Euclidean distance) to the point given
    in parameter point.
    :param points: List of points with each point being a tuple containing coordinates
    :param point: Point as a tuple containing two coordinates
    :return: Furthest point to the given point
    """
    furthest_point = None
    biggest_distance = float('-inf')  # Initialize with a very large number
    for p in points:
        distance = math.dist(p, point)
        if distance > biggest_distance:
            biggest_distance = distance
            furthest_point = p
    return furthest_point


def calculate_minimum_spanning_tree():
    """
    Calculates a minimum spanning tree based on the vertices in verticesStored.
     Also, verticesStored has to be set.
    :return: The minimum spanning tree (As a NetworkX Graph)
    """
    edges = calculate_weighted_edges()
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    mst = nx.minimum_spanning_tree(G)
    return mst


def calculate_minimum_spanning_tree_with_edge(first_index, second_index):
    """
    Calculates a minimum spanning tree based on the vertices in verticesStored, but
    also ensures that the arc between the two vertices with indexes first_index and
    second_index is included. Also, verticesStored has to be set.
    :param first_index: Index of the first vertice of the arc
    :param second_index: Index of the second vertice of the arc
    :return: The minimum spanning tree containing the arc (As a NetworkX Graph)
    """
    edges = calculate_weighted_edges()
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    required_edge = (first_index, second_index)
    # Check if the required edge is in the graph
    if required_edge in G.edges or (required_edge[1], required_edge[0]) in G.edges:
        # Decrease the weight of the required edge to ensure it is included in the MST
        G[required_edge[0]][required_edge[1]]['weight'] = -1
    mst = nx.minimum_spanning_tree(G)
    return mst


def calculate_weighted_edges():
    """
    Calculates the Euclidean arc length for each vertice to each other vertice in
    verticesStored. Also, verticesStored has to be set.
    :return: A list of tuples with each tuple containing three elements: The indexes of the two vertices that are connected
     by the arc and the length of the arc.
    """
    edges = []
    for i in range(len(verticesStored)):
        for j in range(len(verticesStored)):
            if i != j:
                edges.append((i, j, math.dist(verticesStored[i], verticesStored[j])))
    return edges


def calculate_alpha_nearness(first_index, second_index):
    """
    Calculates the alpha-nearness between the two vertices with the indexes. This is the difference between the sum of
    weights of the minimum spanning tree and the sum of weights of the minimum spanning tree containing the arc connecting
    the two vertices. Also, verticesStored has to be set.
    :param first_index: Index of the first vertice
    :param second_index: Index of the second vertice
    :return: alpha-nearness between the two vertices
    """
    mst = calculate_minimum_spanning_tree()
    mstWithEdge = calculate_minimum_spanning_tree_with_edge(first_index, second_index)
    mstWeight = sum(edge[2]['weight'] for edge in mst.edges(data=True))
    mstWithEdgeWeight = (sum(edge[2]['weight'] for edge in mstWithEdge.edges(data=True))
                         + math.dist(verticesStored[first_index], verticesStored[second_index]) + 1)
    return mstWithEdgeWeight - mstWeight


def generate_random_pruned_array(n, p):
    """
    Creates a np array of the dimensions n x n with p percent of the elements being
    1 and the rest 0
    :param n: size of each of the two dimensions of the array
    :param p: percent of 1s in the array
    :return: numpy array of 1s and 0s of the size n x n
    """
    total_elements = n * n
    num_ones = int(total_elements * p / 100)
    array = np.zeros(total_elements, dtype=int)
    array[:num_ones] = 1
    np.random.shuffle(array)
    array = array.reshape((n, n))
    return array

def calculate_distance_array(vertices):
    """
    Calculates the Euclidean distances between each of the vertices in parameter vertices.
    :param vertices: A list of vertices as tuples of coordinates
    :return: A n x n numpy array with the distances between each and every vertice
    """
    vertice_array_1 = np.array(vertices)
    vertice_array_2 = np.array(vertices)
    return np.linalg.norm(vertice_array_1[:, np.newaxis, :] - vertice_array_2[np.newaxis, :, :], axis=2)


def calculate_neighborhood_array(vertices, distances):
    """
    For each vertice in vertices this calculates the neighborhood. That is it sorts all other vertices based on their
    distance to the vertice of interest.
    :param vertices: Numpy array of dimensions n x 2 with n being the number of vertices and 2 being the two coordinates
    :param distances: Numpy array of dimensions n x n with n being the number of vertices. Each element in the array represents
     the distance from the vertice with index of the row index of the element and the vertice with index of the column index
     of the element.
    :return: A numpy array of dimensions n x n x 2. Each row consists of all vertices sorted by their vicinity to the vertice
    with the index that is equal to the row index.
    """
    sorted_indices = np.argsort(distances, axis=1)
    return vertices[sorted_indices]


def instance_size(instance_path):
    """
    Returns the size (=the amount of vertices in an instance), the amount of pickup locations, the amount of origin depots,
    the amount of destination depots and the amount of charging stations for the instance given in instance_path.
    :param instance_path: Path to the instance of interest
    :return: A dictionary with elements:
        "n" - number of vertices;
        "n_P" - number of requests;
        "n_O" - number of origin depots;
        "n_D" - number of destination depots;
        "n_S" -  number of charging stations;
    """
    instance = read_instance(instance_path)
    n = len(instance["vertices"])
    n_O = len(instance["starting_points"])
    n_D = len(instance["destination_points"])
    n_S = len(instance["charging_stations"])
    n_P = int((n - n_O - n_D - n_S)/2)
    instance_sizes = {"n": n,
                      "n_P": n_P,
                      "n_O": n_O,
                      "n_D": n_D,
                      "n_S": n_S}
    return instance_sizes


def read_instance(instance_path):
    """
    Reads file and extracts the data in the instance given at instance_path.
    :param instance_path: Path to the instance of interest
    :return: A dictionary containing elements:
        "vertices" - all vertices (as tuples of coordinates);
        "starting_points" - origin depots (as tuples of coordinates);
        "destination_points" - destination depots (as tuples of coordinates);
        "charging_stations" - charging stations (as tuples of coordinates);
    """
    with open(instance_path, 'r') as file:
        contents = file.readlines()
        nodes = contents[1:len(contents) - 13]
    instance = {"vertices": extract_vertices(nodes),
                "starting_points": extract_startingPoints(contents),
                "destination_points": extract_endingPoints(contents),
                "charging_stations": extract_charging_stations(contents)}
    return instance


def read_solution(solution_path):
    """
    Reads file and extracts for each route each index of the vertices in the route in the solution (based on .sol file)
     given at solution_path.
    :param solution_path: Path to the solution of interest
    :return: A list of routes containing all routes of the .sol file at solution_path
    """
    routes_indices = []
    with open(solution_path, 'r') as file:
        contents = file.readlines()
        routes = contents[1:len(contents) - 1]
    for route in routes:
        vertice_indices = extract_route(route)
        routes_indices.append(vertice_indices)
    return routes_indices