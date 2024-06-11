import numpy as np
import math as math
import networkx as nx
verticesStored = np.array([])


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
    startingPointsIndex.append(startingPointIndex)
    startingPoints = []
    for i in range(len(startingPointsIndex)):
        startingPoints.append(verticesStored[i])
    return startingPoints


def extract_endingPoints(instance):
    endingPointIndex = instance[len(instance) - 12]
    endingPointsIndex = [eval(i) for i in instance[len(instance) - 10].split()]
    endingPointsIndex.append(endingPointIndex)
    endingPoints = []
    for i in range(len(endingPointsIndex)):
        endingPoints.append(verticesStored[i])
    return endingPoints

def extract_ChargingStations(instance):
    chargingStationIndex = instance[len(instance) - 9].split()
    chargingStations = []
    for i in range(len(chargingStationIndex)):
        chargingStations.append(verticesStored[i])
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
            verticeIndeces.append(int(s[0:len(s) - 2]))
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
