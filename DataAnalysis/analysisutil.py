import util
import numpy as np

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
    time_windows = util.extract_timeWindow(filename)
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



