import os

import util
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

########################################################################################################################
##
##                      -- DATA ANALYSIS UTIL --
##  Varying utility to support data analysis like extracting data from files or calculating
##  necessary information.
##
########################################################################################################################

def extract_route_vertice_dims(routes_indices):
    return 0

def extract_route_vertice_coordinates(solution_path, instance_path):
    """
    Extract the vertice coordinates of all routes of the given solution and stores it in
    an n x m array with n being the number of vertices and m being the number of routes.
    :param solution_path: File path to solution has to be a .sol file containing the routes
    :param instance_path: File path to instance has to hold the necessary data for the coordinates of the vertices
    """
    solution_file = util.read_solution(solution_path)
    instance_file = util.read_instance(instance_path)
    vertices = instance_file["vertices"]
    vertices_np = np.array(vertices)
    routes_vertices = []
    for route in solution_file:
        #Adapted list. vertice indices start with 1 but array acces starts with 0
        adapted_route = [x - 1 for x in route]
        route_vertices = vertices_np[adapted_route]
        routes_vertices.append(route_vertices)
    return routes_vertices

def extract_route_development(solution_path):
    route_development_obj = []
    route_development_times = []

    shouldBeRead = False
    with open(solution_path, 'r') as file:
        contents = file.readlines()
        for conent in contents:
            if "\n" == conent and shouldBeRead:
                shouldBeRead = False
            if shouldBeRead:
                obj_value = float(conent.split()[4])
                time_value = float(conent.split()[5])
                if (obj_value < 100000):
                    route_development_times.append(time_value)
                    route_development_obj.append(obj_value)
            if "I       iter             best          obj_old          obj_new        time              method info" in conent:
                shouldBeRead = True
    return route_development_obj, route_development_times


def extract_route_vertice_coordinates_and_time_windows(solution_path, instance_path, filename):
    """
    Creates a python list with each element holding 4 values: the first two are the coordinates,
    the third and fourth are earliest and latest possible service times.
    :param solution_path: Path to the solution of the instance
    :param instance_path: Path to the instance
    :param filename: Name of the file that stores the tightened windows from the deterministic preprocessing
    :return: A list of numpy arrays as list of routes. Each numpy array (=route) contains for each vertice in the route
        Both the coordinates and the earliest and latest service time. Dimensions of each numpy array are n x 4 with n
        being the number of vertices in the route.
    """
    solution_file = util.read_solution(solution_path)
    instance_file = util.read_instance(instance_path)
    vertices = instance_file["vertices"]
    time_windows = util.extract_timeWindow("../tightenedWindows/" + filename)
    vertices_np = np.array(vertices)
    time_windows_np = np.array(time_windows)
    routes_vertices = []
    for route in solution_file:
        # Adapted list. vertice indices start with 1 but array access starts with 0
        adapted_route = [x - 1 for x in route]
        route_vertices = np.zeros((len(adapted_route), 4))
        route_vertices[:, 0:2] = vertices_np[adapted_route]
        route_vertices[:, 2:4] = time_windows_np[adapted_route]
        routes_vertices.append(route_vertices)
    return routes_vertices


def min_max_scale_route(route, vertex=None):
    """
    Scales both the coordinates and time windows of the route and optionally the vertex using Min-Max scaling.
    :param route: ndarray of dimensions n x 4 containing vertices  with columns: [x coordinate, y coordinate ,
     earliest possible service time, latest possible service time]
    :param vertex: (optional) ndarray with new vertex to insert with dimensions 1 x 4
    :return: the scaled route (combined with the vertex if it is set)
    """
    scaler = MinMaxScaler()
    if vertex is not None:
        combined_data = np.vstack([route, vertex])
    else:
        combined_data = route
    scaled_combined = scaler.fit_transform(combined_data)
    return scaled_combined


def calculate_insertion_point_with_scaling(route, vertex):
    """
    Calculate the best insertion point of a vertex into a route based on minimal combined distance between scaled
    coordinates and time windows.
    :param route: ndarray of remaining vertices (n x 4) with columns: [x coordinate, y coordinate , earliest possible service time,
     latest possible service time]
    :param vertex: ndarray of the new vertex to insert (1 x 4) with the same format
    :return: index where the vertex should be inserted based on minimal distance
    """
    scaled_combined_route = min_max_scale_route(route, vertex)
    scaled_route = scaled_combined_route[:-1]
    scaled_vertex = scaled_combined_route[-1]
    route_coords = scaled_route[:, :2]  # (n x 2) Scaled coordinates
    route_times = scaled_route[:, 2:]  # (n x 2) Scaled time windows
    vertex_coords = scaled_vertex[:2]  # (1 x 2) Scaled coordinates
    vertex_times = scaled_vertex[2:]  # (1 x 2) Scaled time windows
    coord_distances = cdist([vertex_coords], route_coords).flatten()
    time_distances = np.abs(route_times - vertex_times).sum(axis=1)
    total_distances = coord_distances + time_distances
    return np.argmin(total_distances)

def execute_evaluation_all_files_sol(function):
    """
    execute the function given in function for each first .sol file of the l1 instances.
    :param function: the function has to take three arguments: The path to a solution, the path to an instance and the
     path to the tightened time windows of the corresponding instance.
    """
    directories_path = "../out/"
    directories = os.listdir(directories_path)
    for directory in directories:
        files_path = os.path.join(directories_path, directory)
        files = os.listdir(files_path)
        instance_path = "../l1/" + directory + ".txt"
        first_sol_file = next((file for file in files if file.endswith('.sol')), None)
        if first_sol_file is not None:
            #print("\n##################################################################################################")
            #print(f"Execute function for {directory}")
            #print("##################################################################################################")
            solution_path = os.path.join(files_path, first_sol_file)
            time_windows_path = directory + ".txt"
            function(solution_path, instance_path, time_windows_path)

