import cv2
import numpy as np
from dlib import full_object
from typing import List, Tuple
#### NOTE any coords below is actually array indices

def get_fiducial_landmarks(image: List[List]) -> List[Tuple]:
    """
    Confirm no. of points for different faces are same
    make sure to convert opencv convention to numpy convention
    """
    pass


def append_zeros_to_landmarks(landmarks: List[Tuple]) -> List[Tuple]:
    """
    reform landmarks: append zeroes(3x2)
    for landmark vector of size nx2, output is (n+3)x2
    """
    pass


def homogenize_coords(coords: List[Tuple]) -> List[Tuple]:
    """
    nx2 -> nx3
    """
    pass


def get_rbf(points1: Tuple, points2: Tuple) -> float:
    """
    1x2, 1x2 -> 1x1
    """
    x1,y1 = points1
    x2,y2 = points2
    norm = abs(y2 - y1) + abs(x2-x1)
    norm_sq = norm**2
    return norm_sq*np.log(norm_sq)


def get_k_matrix(landmarks: List[Tuple],coords: List[Tuple]) -> List[List]:
    """
    landmarks - nx2
    coords - mx2
    output k matrix - mxn
    """
    k = np.zeroes(shape=(len(coords), len(landmarks)))
    for i, coord in enumerate(coords):
        for j, landmark in enumerate(landmarks):
            k[i, j] = get_rbf(coord, landmark)
    return k


def get_tps_matrix(landmarks: List[Tuple], coords: List[Tuple]) -> List[List[int]]:
    """
    nx2 -> (n+3) x (n+3)
    """
    k = get_k_matrix(landmarks)  # nxn
    p = homogenize_coords(coords)  # nx3

    tps1 = np.hstack((k, p))
    tps2 = np.hstack((p.T, np.zeros((3, 3))))
    tps = np.vstack((tps1, tps2))
    return tps


def get_tps_parameters(landmarks_a: List[Tuple], landmarks_b: List[Tuple], lmbda: float = 1e-3) -> List:
    """
    landmarks_b are warped to landmarks_a
    dim(landmarks_a) = dim(landmarks_b) = nx2
    dim(output): (n+3)x2 (parameters for both x and y)
    """
    rhs = append_zeros_to_landmarks(landmarks_a)
    tps = get_tps_matrix(landmarks_b,landmarks_b)
    tps = tps + lmbda * np.identity(len(landmarks_b)+3)
    params = np.dot(np.linalg.inv(tps), rhs)
    return params


def get_loc_in_a(landmarks_b: List[Tuple], point_in_b: Tuple, tps_params) -> Tuple:
    """
    dim(landmarks_b): (n+3)x2
    dim(point_in_b): 1x2
    dim(tps_params): (n+3)x2
    dim(output): 1x2
    """

def construct_img(landmarks_b: List[Tuple], 
                    parameters: List[Tuple], 
                    image_a: List[List[Tuple]],
                    img_shape: Tuple):
    """
    landmarks_b - nx2
    parameters - (n+3)x2
    image_a - k
    img_shape - 1x2
    """
    x_values = np.arange(0,img_shape.shape[1]+1,1)
    y_values = np.arange(0,img_shape.shape[0]+1,1)
    x, y = np.meshgrid(x_values,y_values)
    indices_in_b = np.array([x.flatten(),y.flatten()]).T
    k = get_tps_matrix(landmarks_b,indices_in_b) # mxn
    indices_in_a = k @ parameters
    """
    problems
    1: indices_in_a is maynot be integer
        bilinear interploation?
    2: patch shapes are not same so indices from B might not map within image A
        should we add boundaries as control points
    """
    image_b_warped = np.zeros(shape=(img_shape.shape[0],img_shape.shape[1],3))
    image_b_warped[indices_in_b[:,0],indices_in_bi[:,1],:] = image_a[indices_in_a[:,0],indices_in_a[:,1],:]
    return image_b_warped
    
def main():
    # Read image/s
    # Save crops of faces in image frame
    # Save image_rects (location of crop wrt the original image/frame)
    # For both crops
        # Get facial landmarks
        # Get tps parameters for image
        # Get (x, y) for a
        # Construct img using (x, y)
    # Get inverse warpings for both crops
    # Put warped crops on original image/frame
    # Display
    pass


if __name__ == '__main__':
    main()
