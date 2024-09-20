import numpy as np
import math as math
import networkx as nx
import os
verticesStored = np.array([])
timeWindowsStored = np.array([])
infeasible_arcs_stored = np.array([])

def calculate_centroid(points):
    # Initialize sums
    sum_x = 0
    sum_y = 0

    # Number of points
    n = len(points)

    # Sum all x and y coordinates
    for i in range(len(points)):
        sum_x += points[i][0]
        sum_y += points[i][1]

    # Calculate the centroid
    centroid_x = sum_x / n
    centroid_y = sum_y / n

    return centroid_x, centroid_y


def extract_startingPoints(instance):
    startingPointIndex = instance[len(instance) - 13]
    startingPointsIndex = [eval(i) for i in instance[len(instance) - 11].split()]
    startingPointsIndex.append(eval(startingPointIndex))
    startingPoints = []
    for i in range(len(startingPointsIndex)):
        startingPoints.append(verticesStored[int(startingPointsIndex[i])])
    return startingPoints

def extract_first_startingPoints_index(instance):
    return int(instance[len(instance) - 13])

def extract_endingPoints(instance):
    endingPointIndex = instance[len(instance) - 12]
    endingPointsIndex = [eval(i) for i in instance[len(instance) - 10].split()]
    endingPointsIndex.append(eval(endingPointIndex))
    endingPoints = []
    for i in range(len(endingPointsIndex)):
        endingPoints.append(verticesStored[int(endingPointsIndex[i])])
    return endingPoints

def extract_ChargingStations(instance):
    chargingStationIndex = instance[len(instance) - 9].split()
    chargingStations = []
    for i in range(len(chargingStationIndex)):
        chargingStations.append(verticesStored[int(chargingStationIndex[i])-1])
    return chargingStations

def extract_objectiveFun(solution):
    objectiveFunction = find_lines_with_substring(solution, "T best obj:")
    if (len(objectiveFunction) != 1):
        return -1
    return float(objectiveFunction[0].split()[3])


def extract_vertices(instance):
    vertices = []
    for each in instance:
        preCoord = each.split()
        xCoord = float(preCoord[1])
        yCoord = float(preCoord[2])
        vertices.append((xCoord, yCoord))
    global verticesStored
    verticesStored = vertices
    return vertices

def extract_timeWindow(instance):
    timeWindow = []
    file_path_time_windows = os.path.join("tightenedWindows/", instance)
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

def extract_infeasible_arcs(instance):
    infeasible_arcs = []
    file_path_infeasible_arcs = os.path.join("infeasibleArcs/", instance)
    with open(file_path_infeasible_arcs, 'r') as file:
        for line in file:
            values = line.split(" ")
            infeasible_arcs.append([1 if value == "true" else 0 for value in values[0:-1]])
    global infeasible_arcs_stored
    infeasible_arcs_stored = infeasible_arcs
    print(np.array(infeasible_arcs).shape)
    return np.array(infeasible_arcs).flatten()

def find_lines_with_substring(solution, substring):
    result = []
    for line in solution:
        if substring in line:
            result.append(line)
    return result


def extract_route(route):
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

def findSmallestDistance(points, point):
    closest_point = None
    smallest_distance = float('inf')  # Initialize with a very large number

    for p in points:
        distance = math.dist(p, point)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_point = p

    return closest_point
def findBiggestDistance(points, point):
    furthest_point = None
    biggest_distance = float('-inf')  # Initialize with a very large number

    for p in points:
        distance = math.dist(p, point)
        if distance > biggest_distance:
            biggest_distance = distance
            furthest_point = p
    return furthest_point

def calculateMinimumSpanningTree():
    edges = calculateWeightedEdges()
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    mst = nx.minimum_spanning_tree(G)
    return mst

def calculateMinimumSpanningTreeWithEdge(firstIndex, secondIndex):
    edges = calculateWeightedEdges()
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    required_edge = (firstIndex, secondIndex)
    # Check if the required edge is in the graph
    if required_edge in G.edges or (required_edge[1], required_edge[0]) in G.edges:
        # Decrease the weight of the required edge to ensure it is included in the MST
        G[required_edge[0]][required_edge[1]]['weight'] = -1
    mst = nx.minimum_spanning_tree(G)
    return mst

def calculateWeightedEdges():
    edges = []
    for i in range(len(verticesStored)):
        for j in range(len(verticesStored)):
            if i != j:
                edges.append((i, j, math.dist(verticesStored[i], verticesStored[j])))
    return edges

def calculatealphaNearness(firstIndex, secondIndex):
    mst = calculateMinimumSpanningTree()
    mstWithEdge = calculateMinimumSpanningTreeWithEdge(firstIndex, secondIndex)
    mstWeight = sum(edge[2]['weight'] for edge in mst.edges(data=True))
    mstWithEdgeWeight = sum(edge[2]['weight'] for edge in mstWithEdge.edges(data=True)) + math.dist(verticesStored[firstIndex], verticesStored[secondIndex]) + 1
    return mstWithEdgeWeight - mstWeight
def generate_random_pruned_array(n, p):
    total_elements = n * n
    num_ones = int(total_elements * p / 100)
    array = np.zeros(total_elements, dtype=int)
    array[:num_ones] = 1
    np.random.shuffle(array)
    array = array.reshape((n, n))
    return array
def calculateDistanceArray(vertices):
    vertice_array_1 = np.array(vertices)
    vertice_array_2 = np.array(vertices)
    return np.linalg.norm(vertice_array_1[:, np.newaxis, :] - vertice_array_2[np.newaxis, :, :], axis=2)

def calculateNeighborhoodArray(vertices, distances):
    sorted_indices = np.argsort(distances, axis=1)
    return vertices[sorted_indices]

"""
    Extract the vertice coordinates of all routes of the given solution an stores it in
    an n x m array with n being the number of vertices and m being the number of routes. 
    
    :parameter file path to solution has to be a .sol file containg the routes
    :parameter file path to instance has to hold the necessary data for the coordinates of the vertices
"""
def extract_route_vertice_coordinates(solution, instance):
    solution_file = read_solution(solution)
    instance_file = read_instance(instance)
    vertices = instance_file["vertices"]
    vertices_np = np.array(vertices)
    routes_vertices = []
    for route in solution_file:
        #Adapted list. vertice indices start with 1 but array acces starts with 0
        adapted_route = [x - 1 for x in route]
        route_vertices = vertices_np[adapted_route]
        routes_vertices.append(route_vertices)
    return routes_vertices


"""
    Extracts the date in the instance given at instance_path. 
"""
def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        contents = file.readlines()
        nodes = contents[1:len(contents) - 13]
    instance = {"vertices": extract_vertices(nodes),
                "starting_points": extract_startingPoints(contents)}
    return instance


"""
    Extracts a list of routes from a .sol file given in solution_path. 
"""
def read_solution(solution_path):
    routes_indices = []
    with open(solution_path, 'r') as file:
        contents = file.readlines()
        routes = contents[1:len(contents) - 1]
    for route in routes:
        verticeIndices = extract_route(route)
        routes_indices.append(verticeIndices)
    return routes_indices

"""
    Returns a python list with each element holding 4 values: the first two are the coordinates, 
    the third and fourth are earliest and latest possible service times. 
"""
def extract_route_vertice_coordinates_and_time_windows(solution, instance, filename):
    solution_file = read_solution(solution)
    instance_file = read_instance(instance)
    vertices = instance_file["vertices"]
    time_windows = extract_timeWindow(filename)
    vertices_np = np.array(vertices)
    time_windows_np = np.array(time_windows)
    routes_vertices = []
    for route in solution_file:
        # Adapted list. vertice indices start with 1 but array acces starts with 0
        adapted_route = [x - 1 for x in route]
        route_vertices = np.zeros((len(adapted_route), 4))
        hello = route_vertices[:, 2:4]
        test = time_windows_np[adapted_route]
        route_vertices[:, 0:2] = vertices_np[adapted_route]
        route_vertices[:, 2:4] = time_windows_np[adapted_route]
        routes_vertices.append(route_vertices)
    return routes_vertices
