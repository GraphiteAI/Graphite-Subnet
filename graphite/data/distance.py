from typing import Iterable, Union
import bittensor as bt
from geopy.distance import geodesic

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

def geom(lat_lon_1:Iterable[Union[int, float]], lat_lon_2:Iterable[Union[int, float]], ellipsoid="WGS-84"):
    '''
    default settings for geom compute using WGS-84 format for coordinate values.
    Refer to "https://geopy.readthedocs.io/en/stable/#module-geopy.distance" for documentation of different ellipsoid representations
    
    function requires the input values to be in (lat, lon) format.
    '''
    try:
        geodesic(lat_lon_1, lat_lon_2, ellipsoid=ellipsoid).meters
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