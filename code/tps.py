import cv2
import numpy as np
from dlib import full_object
from typing import List, Tuple


def get_fiducial_landmarks(image: List[List]) -> List[Tuple]:
    """
    Confirm no. of points for different faces are same
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
    norm = abs(points1 - points2)
    norm_sq = norm**2
    return norm_sq*np.log(norm_sq)


def get_k_matrix(points: List[Tuple]) -> List[List]:
    """
    nx2 -> nxn
    """
    k = np.zeroes(shape=(len(points), len(points)))
    for i, point_i in enumerate(points):
        for j, point_j in enumerate(points):
            k[i, j] = get_rbf(point_i, point_j)
    return k


def get_tps_matrix(landmarks: List[Tuple]) -> List[List[int]]:
    """
    nx2 -> (n+3) x (n+3)
    """
    k = get_k_matrix(landmarks)  # nxn
    p = homogenize_coords(landmarks)  # nx3

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
    tps = get_tps_matrix(landmarks_b)
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


def construct_img():
    pass


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
