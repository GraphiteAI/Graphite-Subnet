from typing import Iterable, Union
import bittensor as bt
# from geopy.distance import geodesic
import math
import numpy as np

REF_PI = 3.141592
REF_EARTH_RADIUS = 6378.388

### Defining distance functions
def euc_2d(head:Iterable[Union[int,float]], tail:Iterable[Union[int,float]]):
    '''
    Computes the 2d euclidean distance of head vs tail
    '''
    try:
        return ((head[0]-tail[0])**2 + (head[1]-tail[1])**2)**0.5
    except IndexError as e:
        bt.logging.error("Input node structure not suitable for Euclidean 2D: {e}")

def euc_3d(head:Iterable[Union[int,float]], tail:Iterable[Union[int,float]]):
    '''
    Computes the 3d euclidean distance of head vs tail
    '''
    try:
        return ((head[0]-tail[0])**2 + (head[1]-tail[1])**2 + (head[2]-tail[2])**2)**0.5
    except IndexError as e:
        bt.logging.error(f"Input node structure not suitable for Euclidean 2D: {e}")

def geom(lat_lon_1:Iterable[Union[int, float]], lat_lon_2:Iterable[Union[int, float]]):
    '''
    default settings for geom compute using WGS-84 format for coordinate values.
    Refer to "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/TSPFAQ.html" for geom dist definition
    
    function requires the input values to be in (lat, lon) format.
    '''
    try:
        q1 = math.cos(lat_lon_1[1] - lat_lon_2[1])
        q2 = math.cos(lat_lon_1[0] - lat_lon_2[0])
        q3 = math.cos(lat_lon_1[0] + lat_lon_2[0])
        return REF_EARTH_RADIUS * (math.acos( 0.5*((1.0+q1)*q2 - (1.0-q1)*q3) ) + 1.0)
    except ValueError as e:
        bt.logging.error(f"Invalid lat/lon values: {e}")

def man_2d(head:Iterable[Union[int,float]], tail:Iterable[Union[int,float]]):
    '''
    Computes the 2d manhattan distance of head vs tail
    '''
    try:
        return abs(head[0]-tail[0]) + abs(head[1]-tail[1])
    except IndexError as e:
        bt.logging.error("Input node structure not suitable for Euclidean 2D: {e}")

def man_3d(head:Iterable[Union[int,float]], tail:Iterable[Union[int,float]]):
    '''
    Computes the 3d manhattan distance of head vs tail
    '''
    try:
        return abs(head[0]-tail[0]) + abs(head[1]-tail[1]) + abs(head[2]-tail[2])
    except IndexError as e:
        bt.logging.error("Input node structure not suitable for Manhattan 3D: {e}")

def geom_edges(lat_lon_array):
    '''
    Vectorized geom distance calculation using numpy.
    Requires lat_lon_array to be an Nx2 array where N is the number of nodes, 
    and the columns represent latitude and longitude respectively.
    '''
    lat_lon_array = np.deg2rad(lat_lon_array)
    # Split latitudes and longitudes into separate arrays
    lat = lat_lon_array[:, 0]  # All latitudes
    lon = lat_lon_array[:, 1]  # All longitudes

    # Broadcast for pairwise distance computation
    lat1, lat2 = np.meshgrid(lat, lat)  # Pairwise latitudes
    lon1, lon2 = np.meshgrid(lon, lon)  # Pairwise longitudes

    # Calculate q1, q2, q3 using numpy's vectorized functions
    q1 = np.cos(lon1 - lon2)
    q2 = np.cos(lat1 - lat2)
    q3 = np.cos(lat1 + lat2)

    # Compute distances in a vectorized way
    distances = REF_EARTH_RADIUS * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0

    # Ensure the diagonal (distance between the same nodes) is 0
    np.fill_diagonal(distances, 0)

    return distances


def man_2d_edges(lat_lon_array: np.ndarray) -> np.ndarray:
    '''
    Vectorized Manhattan distance calculation using numpy.
    Requires lat_lon_array to be an Nx2 array where N is the number of nodes, 
    and the columns represent latitude and longitude respectively.
    '''
    # Split latitudes and longitudes into separate arrays
    lat = lat_lon_array[:, 0]  # All latitudes
    lon = lat_lon_array[:, 1]  # All longitudes

    # Broadcast for pairwise distance computation
    lat1, lat2 = np.meshgrid(lat, lat)  # Pairwise latitudes
    lon1, lon2 = np.meshgrid(lon, lon)  # Pairwise longitudes

    # Calculate Manhattan distances in a vectorized way
    distances = np.abs(lat1 - lat2) + np.abs(lon1 - lon2)

    # Ensure the diagonal (distance between the same nodes) is 0
    np.fill_diagonal(distances, 0)

    return distances

def euc_2d_edges(lat_lon_array: np.ndarray) -> np.ndarray:
    '''
    Vectorized Euclidean distance calculation using numpy.
    Requires lat_lon_array to be an Nx2 array where N is the number of nodes, 
    and the columns represent latitude and longitude respectively.
    '''
    # Split latitudes and longitudes into separate arrays
    lat = lat_lon_array[:, 0]  # All latitudes
    lon = lat_lon_array[:, 1]  # All longitudes

    # Broadcast for pairwise distance computation
    lat1, lat2 = np.meshgrid(lat, lat)  # Pairwise latitudes
    lon1, lon2 = np.meshgrid(lon, lon)  # Pairwise longitudes

    # Calculate Euclidean distances in a vectorized way
    distances = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

    # Ensure the diagonal (distance between the same nodes) is 0
    np.fill_diagonal(distances, 0)

    return distances