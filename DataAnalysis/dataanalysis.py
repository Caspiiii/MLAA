import networkx as nx

import util
import analysisutil as anu
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
from scipy.spatial.distance import cdist

########################################################################################################################
##
##                      -- DATA ANALYSIS --
##  Extract and analyse data of both instances and solutions of the EADARP
##
########################################################################################################################



def route_distribution():
    """
       Plot all routes of each instance in the given directory in terms of their
       geographical coordinates.
    """
    directories_path = "../out/"
    directories = os.listdir(directories_path)
    for directory in directories:
        files_path = os.path.join(directories_path, directory)
        files = os.listdir(files_path)
        instance = "../l1/" + directory + ".txt"
        for filename in files:
            if filename.endswith("sol"):
                plt.figure(figsize=(10, 10))
                plt.title(f"Instance " + instance.split("/")[1])
                file_path = os.path.join(files_path, filename)
                routes_vertices = anu.extract_route_vertice_coordinates(file_path, instance)
                draw_furthest_quadrilateral(routes_vertices, instance)
                plt.grid(True)
                plt.savefig(f"{filename[:-4]}_locations_all_routes.png", dpi=300, bbox_inches='tight')  # Save the figure to a file
                plt.show()
                break



def order_by_closest(furthest_points):
    """
        Helper function that takes the vertices furthest away from each other and sorts
        them based on their vicinity to the vertice before.
    """
    furthest_points = np.array(furthest_points)

    # Start with the first point (you can choose any point to start with)
    ordered_points = [furthest_points[0]]
    remaining_points = np.copy(furthest_points[1:])

    # Greedily pick the next closest point
    while remaining_points.shape[0] > 0:
        last_point = ordered_points[-1]
        # Compute distances from the last point to all remaining points
        distances = np.linalg.norm(remaining_points - last_point, axis=1)
        # Find the index of the closest point
        closest_index = np.argmin(distances)
        # Add the closest point to the ordered list
        closest_point = remaining_points[closest_index]
        ordered_points.append(closest_point)
        # Remove the closest point from the remaining points
        remaining_points = np.delete(remaining_points, closest_index, axis=0)
    return np.array(ordered_points)



def draw_furthest_quadrilateral(routes_vertices, name):
    """
        Plots both the geographical location of all vertices for the given instance based on
        their corresponding route and additionally draws a quadrilateral that encases (most)
        vertices of the corresponding route.
    """
    cmap = mpl.colormaps['tab10']
    colors = cmap(np.linspace(0, 1, len(routes_vertices)))
    for i, vertices in enumerate(routes_vertices):
        color = colors[i]  # Get the color for this plot
        #plt.subplot(math.ceil(math.sqrt(len(routes_vertices))),
        #            math.floor(math.sqrt(len(routes_vertices) + 2)),
        #            i + 1)
        #plt.scatter(np.array(util.verticesStored)[:, 0], np.array(util.verticesStored)[:, 1], s=5, color="black")
        if len(vertices) < 4:
            print(f"Not enough vertices in ndarray {i + 1} to form a quadrilateral")
            continue

        max_distance_sum = 0
        furthest_points = None

        # Get all combinations of 4 points to form a quadrilateral
        for points in combinations(vertices, 4):
            # Calculate total distance between all pairs of the 4 selected points
            distance_sum = sum(np.linalg.norm(p1 - p2) for p1, p2 in combinations(points, 2))

            # Update if this combination has the largest total distance
            if distance_sum > max_distance_sum:
                max_distance_sum = distance_sum
                furthest_points = np.array(points)

        # Draw the area (quadrilateral) between the four furthest apart points
        # Plot the points of the current route
        plt.scatter(vertices[:, 0], vertices[:, 1], color=color, label="Vertices")

        # Plot the quadrilateral if the furthest points were found
        #if furthest_points is not None:
        # Close the polygon by repeating the first point at the end
        #    furthest_points_ordered = order_by_closest(furthest_points)
        #    quadrilateral = np.vstack([furthest_points_ordered, furthest_points_ordered[0]])

        # Draw the quadrilateral (polygon)
        #    plt.plot(quadrilateral[:, 0], quadrilateral[:, 1], color=color, label="Furthest quadrilateral")
        #plt.legend()



def pca_analysis_all_routes(routes_vertices):
    """
        Helper function to additionally generate and plot PC1 and PC2 for all other vertices of
        the instance.
    """
    scaler = StandardScaler()
    for i, route_vertices in enumerate(routes_vertices):
        scaled_data = scaler.fit_transform(route_vertices)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        plt.scatter(pca_df['PC1'], pca_df['PC2'], s=10, color="black")


def pca_analysis():
    """
        Execute PCA Analysis based on currently 4 features: The first two are the location coordinates.
        The third and fourth are the earliest and latest possible service time.
    """
    directories_path = "/home/caspiiii/TU_Wien/Bachelor/data/24h/"
    #directories = os.listdir(directories_path)[3:]

    #for directory in directories:
    #
    #plt.xlabel('Principal Component 1')
    #plt.ylabel('Principal Component 2')
    #files_path = os.path.join(directories_path, directory)
    #files = os.listdir(files_path)
    files = os.listdir(directories_path)
    #instance = "../l1/" + directory + ".txt"
    for filename in files:
        if filename.endswith("sol"):
            instance = "../l2/" + filename[:-7] + ".txt"
            file_path = os.path.join(directories_path, filename)
            routes_vertices = anu.extract_route_vertice_coordinates_and_time_windows(file_path, instance,
                                                                                      filename[:-7] + ".txt")
            cmap = mpl.colormaps['Set3']
            colors = cmap(np.linspace(0, 1, len(routes_vertices)))
            scaler = StandardScaler()
            for i, route_vertices in enumerate(routes_vertices):
                scaled_data = scaler.fit_transform(route_vertices)
                plt.figure(figsize=(10, 10))
                pca = PCA(n_components=2)  # Reduce to 2 dimensions (components)
                principal_components = pca.fit_transform(scaled_data)
                pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                # TODO: generalize
                #plt.subplot(math.ceil(math.sqrt(len(routes_vertices))),
                #            math.floor(math.sqrt(len(routes_vertices) + 2)), i + 1)
                plt.title('PCA: ' + filename[:-7] + "; Route: " + str(i + 1))
                pca_analysis_all_routes(routes_vertices)
                plt.scatter(pca_df['PC1'], pca_df['PC2'], color=colors[i])
                plt.grid(True)
                explained_variance = pca.explained_variance_ratio_
                print("Explained Variance Ratio:", explained_variance)
                plt.tight_layout()
                plt.savefig(f'pca_{filename[:-7]}_route{i}', dpi=300)
                #plt.show()



def route_distribution_with_annotations():
    """
    Draws all points of the instance based on the route that they were assigned to. Also assign to each vertice if it
    is a pickup-, dropoff- or other location.
    """
    directories_path = "../out/"
    directories = os.listdir(directories_path)
    for directory in directories:
        files_path = os.path.join(directories_path, directory)
        files = os.listdir(files_path)
        instance_path = "../l1/" + directory + ".txt"
        cmap = mpl.colormaps['tab10']
        colors = cmap(np.linspace(0, 1, 5))
        for filename in files:
            if filename.endswith("sol"):
                solution_path = os.path.join(files_path, filename)
                routes_vertices = anu.extract_route_vertice_coordinates(solution_path, instance_path)
                instance_dims = util.instance_size(instance_path)
                # Number of pickup locations
                n_P = instance_dims["n_P"]
                routes_indices = util.read_solution(solution_path)
                # iterate over all routes
                plt.figure(figsize=(10, 10))
                plt.title(f"Instance " + instance_path.split("/")[2])
                plt.axis("off")
                for i, vertices in enumerate(routes_vertices):
                    route_indices = routes_indices[i]
                    counter = 0
                    plt.subplot(math.ceil(math.sqrt(len(routes_vertices))),
                                math.floor(math.sqrt(len(routes_vertices) + 2)), i+1)
                    plt.grid(True)
                    # iterate over one route of all routes
                    for index in route_indices:
                        if index < n_P:
                            plt.scatter(vertices[counter, 0], vertices[counter, 1], s=100, color="#1f77b4", label="Pickup")
                        elif n_P <= index < 2 * n_P:
                            # plot line between pickup and drop-off location
                            all_vertices = util.verticesStored
                            pickup_location = all_vertices[index - n_P - 1]
                            x_coordinates = [pickup_location[0], vertices[counter][0]]
                            y_coordinates = [pickup_location[1], vertices[counter][1]]
                            plt.plot(x_coordinates, y_coordinates, color='lightgrey', linewidth=5, alpha=0.7)
                            plt.scatter(vertices[counter, 0], vertices[counter, 1], s=100,
                                        color="#ff7f0e", label="Drop-off")
                        else:
                            plt.scatter(vertices[counter:, 0], vertices[counter:, 1], s=100,
                                        color="#2ca02c", label="other")
                        if counter != 0:
                            plt.arrow(vertice_before[0], vertice_before[1], vertices[counter][0] - vertice_before[0],
                                    vertices[counter][1] - vertice_before[1], length_includes_head=1, shape="full",
                                    head_width=0.7, head_length=0.7)

                        vertice_before = vertices[counter]
                        counter += 1
                    #plt.legend()
                plt.tight_layout()
                plt.show()


                break


def route_distribution_time_windows():
    """
    For each route over all solutions plot the time windows of each pickup- and drop-off location.
    """
    directories_path = "/home/caspiiii/TU_Wien/Bachelor/data/24h/"
    files = os.listdir(directories_path)
    """        
    directories_path = "../out/"
    directories = os.listdir(directories_path)
    for directory in directories:
        files_path = os.path.join(directories_path, directory)
        files = os.listdir(files_path)
        instance_path = "../l1/" + directory + ".txt"
        for filename in files:
            if filename.endswith("sol"):
            """
    for filename in files:
        if filename.endswith("sol"):
            instance_path = "../l2/" + filename[:-7] + ".txt"
            #solution_path = os.path.join(files_path, filename)
            solution_path = os.path.join(directories_path, filename)
            routes_vertices = anu.extract_route_vertice_coordinates_and_time_windows(solution_path, instance_path, filename[:-7] + ".txt")
            #num_plots = len(routes_vertices)
            #fig, axs = plt.subplots(num_plots, 1, figsize=(20, num_plots * 3), constrained_layout=True)
            for idx, vertices in enumerate(routes_vertices):
                fig, axs = plt.subplots(1, 1, figsize=(20, 20), constrained_layout=True)
                ax = axs
                earliest_times = vertices[:, 2]
                latest_times = vertices[:, 3]
                for i, (start, end) in enumerate(zip(earliest_times, latest_times)):
                    ax.barh(i, end - start, left=start, height=0.4, color='skyblue', edgecolor='black')
                #ax.set_yticks(np.arange(len(vertices)))
                ax.set_ylabel('Locations', fontsize=16)
                #ax.set_yticklabels([f"Location {i + 1}" for i in range(len(vertices))], fontsize=6)
                ax.set_xlabel('Time', fontsize=16)
                ax.set_title(f'Execution Time Overlap for Route {idx + 1}', fontsize=16)
                ax.tick_params(axis='x', labelsize=8)
                fig.suptitle(f"Instance {filename[:-7]}", fontsize=20)
                plt.show()
            break


def evaluate_routes_with_scaling(solution_path, instance_path, time_windows_path):
    """
    Evaluate the distance-based heuristic using scaled coordinates and time windows.
    """
    differences = []  # List to store the differences between predicted and actual insertion points
    routes_vertices = anu.extract_route_vertice_coordinates_and_time_windows(solution_path, instance_path,
                                                                             time_windows_path)
    routes_indices = util.read_solution(solution_path)
    n_requests = util.instance_size(instance_path)["n_P"]
    for i in range(len(routes_vertices)):
        route = routes_vertices[i]
        route_indices = routes_indices[i]
        n = len(route)
        for j in range(n):
            if (route_indices[j] > 2 * n_requests):
                continue
            remaining_route = np.delete(route, j, axis=0)
            vertex_to_insert = route[j]
            predicted_insertion_point = anu.calculate_insertion_point_with_scaling(remaining_route, vertex_to_insert)
            actual_insertion_point = j
            difference = abs(predicted_insertion_point - actual_insertion_point)
            differences.append(difference)
            #print(
            #    f"Vertex {j}: Predicted: {predicted_insertion_point}, Actual: {actual_insertion_point}, Difference: {difference}")
    print(f"Average differences over all routes: {np.mean(differences)}")


def drop_off_route_distribution():
    directories_path = "../out/"
    directories = os.listdir(directories_path)
    index_distances = []
    for directory in directories:
        files_path = os.path.join(directories_path, directory)
        files = os.listdir(files_path)
        for filename in files:
            if filename.endswith("sol"):
                file_path = os.path.join(files_path, filename)
                with open(file_path, 'r') as file:
                    # calculate weightings
                    contents = file.readlines()
                    routes = contents[1:len(contents) - 1]
                    routeCounter = 1
                for route in routes:
                    verticeIndices = util.extract_route(route)
                    instance = "../l1/" + directory + ".txt"
                    # the origindepots have the lowest ids bigger than the ids of the actual requests. Therefore taking the
                    # lowest id of the startingpoints gives the amount of pickup and dropoff locations combined.
                    with open(instance, 'r') as file:
                        contents = file.readlines()
                    startingPoints = util.extract_first_startingPoints_index(contents)
                    n = startingPoints - 1
                    # Create pairs between i and i + n//2 (e.g., 1 <-> 6, 2 <-> 7, etc.)
                    half_n = math.floor(n / 2)
                    pairs = [(i, i + half_n) for i in range(1, half_n + 1)]  # Generate pairs like (1, 6), (2, 7), etc.
                    # Find distances between pairs
                    distances = []
                    for pair in pairs:
                        element1, element2 = pair  # e.g. (1, 6), (2, 7), etc.
                        # Check if both elements of the pair are present in the list
                        if element1 in verticeIndices and element2 in verticeIndices:
                            # Find their positions in the list
                            index1 = verticeIndices.index(element1)
                            index2 = verticeIndices.index(element2)
                            # Calculate the number of elements between them
                            dist = abs(
                                index1 - index2) - 1  # Distance between the indices minus 1 (for in-between elements)
                            distances.append(dist)
                        elif element1 not in verticeIndices and element2 not in verticeIndices:
                            # Skip this pair if either element is missing
                            distances.append(None)
                        else:
                            raise Exception("Dis no gud")
                    # Output the distances
                    print("Distances between pairs (None if missing):", distances)
                    # Plotting the distances, excluding missing pairs
                    valid_distances = [dist for dist in distances if dist is not None]
                    """        
                    plt.plot(valid_distances, marker='o')
                    plt.title("Number of Elements Between Pick-up/Drop-off Pairs (" + directory + ", routeNr.: " + str(
                        routeCounter) + ")")
                    plt.xlabel("Valid Pair Index")
                    plt.ylabel("Number of Elements Between")
                    plt.grid(True)
                    plt.show()
                    """
                    index_distances+=valid_distances
                    routeCounter += 1
            break
    mean = np.mean(index_distances)
    variance = np.var(index_distances)
    std_dev = np.std(index_distances)
    median = np.median(index_distances)
    minimum = np.min(index_distances)
    maximum = np.max(index_distances)
    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Median: {median}")
    print(f"Minimum: {minimum}")
    print(f"Maximum: {maximum}")


#drop_off_route_distribution()

def draw_example_graph(nodes):
    G = nx.DiGraph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from((i, j) for i in range(len(nodes)) for j in range(len(nodes)) if i != j)
    node_colors = []
    labels = {}
    for i, (name, color) in enumerate(nodes):
        node_colors.append(color)
        labels[i] = name
    pos = nx.spring_layout(G, center=(0, 0), k=0.1, scale=3.0)
    #forcing first node in center
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        labels=labels,
        node_size=2000,
        font_color="black",
        font_size=16,
        font_weight="bold",
        edge_color="gray",
        width=1.5,
        connectionstyle="arc3,rad=0.05"
    )
    plt.title("Complete Graph with Charging Station Centered", fontsize=14)
    plt.show()

#route_distribution()
#pca_analysis()
#route_distribution_with_annotations()
route_distribution_time_windows()
#anu.execute_evaluation_all_files_sol(evaluate_routes_with_scaling)
nodes = [
        ("Charging Station", "#8fc78f"),
        ("PickUp 1", "#add8e6"),
        ("PickUp 2", "#add8e6"),
        ("Origin Depot", "#fbbd8b"),
        ("DropOff 1", "#add8e6"),
        ("DropOff 2", "#add8e6"),
        ("Destination Depot", "#fbbd8b")
    ]
#draw_example_graph(nodes)
