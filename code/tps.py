import cv2
import numpy as np
import dlib
from typing import List, Tuple, Any
from utils import *
import argparse
#### NOTE any coords below is actually array indices

def append_zeros_to_landmarks(landmarks: List[Tuple]) -> List[Tuple]:
    """
    reform landmarks: append zeroes(3x2)
    for landmark vector of size nx2, output is (n+3)x2
    """
    ret = np.concatenate((landmarks,np.zeros((3,2))),axis=0)
    return ret


def get_k_matrix(landmarks: List[Tuple],coords: List[Tuple]) -> List[List]:
    """
    landmarks - nx2
    coords - mx2
    output k matrix - mxn
    """
    coords_x, landmarks_x = np.meshgrid(landmarks[:,0],coords[:,0])
    coords_y, landmarks_y = np.meshgrid(landmarks[:,1],coords[:,1])

    # rbf_log
    # norm = abs(landmarks_y - coords_y) + abs(landmarks_x - coords_x)
    # norm_sq = norm**2
    # K = norm_sq*np.log10(norm_sq + 1e-6)

    # rbf_gaussian
    norm = abs(landmarks_y - coords_y)**2 + abs(landmarks_x - coords_x)**2
    K = np.exp(-norm/100)
    return K

def get_tps_matrix(landmarks: List[Tuple], coords: List[Tuple], compute_coords: bool) -> List[List[int]]:
    """
    nx2 -> (n+3) x (n+3)
    """
    k = get_k_matrix(landmarks,coords)  # nxn
    p = homogenize_coords(coords)  # nx3

    tps1 = np.hstack((k, p))
    if compute_coords:
        return tps1

    tps2 = np.hstack((p.T, np.zeros((3, 3))))
    tps = np.vstack((tps1, tps2))
    return tps


def get_tps_parameters( landmarks_from: List[Tuple], 
                        landmarks_to: List[Tuple], 
                        lmbda: float = 1e-3) -> List:
    """
    landmarks_from are warped to landmarks_to
    dim(landmarks_from) = dim(landmarks_to) = nx2
    dim(output): (n+3)x2 (parameters for both x and y)
    """

    tps = get_tps_matrix(landmarks_from,landmarks_from,False)

    # adding lambda for stable inverse
    tps = tps + lmbda * np.identity(len(landmarks_from)+3)

    rhs = append_zeros_to_landmarks(landmarks_to)
    params = np.dot(np.linalg.inv(tps), rhs)

    return params


def warp_patch(landmarks_from : List[Tuple],
               parameters     : List[Tuple], 
               frame          : List[List[Tuple]],
               canvas         : List[List[Tuple]],
               patch_rect_to  : Any):
    """
    landmarks_from  - nx2
    parameters      - (n+3)x2
    frame           - hxw
    patch_rect_to   - 4x2
    """

    # Filter coords from patch that lie within the convex hull of landmarks
    mask = np.zeros((frame.shape[0], frame.shape[1]))
    landmarks_from_raw = landmarks_from[:-8].astype(np.int32)
    hull_points_to = cv2.convexHull(landmarks_from_raw, returnPoints=True)
    mask = cv2.fillConvexPoly(mask, hull_points_to, 1)
    indices_from = np.argwhere(mask==1)

    # Convert indices to coords because cv functions above changed the type
    coords_from = np.vstack((indices_from[:,1], indices_from[:,0])).T

    tps_mat = get_tps_matrix(landmarks_from,coords_from, True) # mxn
    coords_to = tps_mat @ parameters
    coords_to = coords_to.astype(int)

    warped_patch_shape = dlib_rect_to_shape(patch_rect_to)
    warped_patch = np.zeros(shape=(warped_patch_shape[0],warped_patch_shape[1],3))

    canvas[coords_from[:,1],coords_from[:,0],:] = frame[coords_to[:,1],coords_to[:,0],:]

    return canvas
    
def main(args):
    display = args.display
    debug = args.debug
    predictor_model = "../data/shape_predictor_68_face_landmarks.dat"

    # Initialize frontal face detector and shape predictor:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)

    # Read image/s
    frame = cv2.imread("../data/selfie_4.jpeg")
    print(frame.shape)
    frame = cv2.resize(frame ,(550,400))
    print(frame.shape)

    # each frame has two faces that needs to be swapped
    rects = detector(frame, 0)
    rect_a = rects[0]
    rect_b = rects[1]

    # Get facial landmarks
    landmarks_a = get_fiducial_landmarks(predictor,frame,rect_a,args,"a")
    landmarks_b = get_fiducial_landmarks(predictor,frame,rect_b,args,"b")

    if args.display:
        img_fiducials = frame.copy()
        draw_fiducials([landmarks_a,landmarks_b],img_fiducials,"ab")

    # Append rect corner and center coords as additional landmarks
    landmarks_a = append_rect_coords_to_landmarks(landmarks_a, rect_a)
    landmarks_b = append_rect_coords_to_landmarks(landmarks_b, rect_b)

    # Get tps parameters for image
    warped_params_of_b = get_tps_parameters(landmarks_b, landmarks_a)
    warped_params_of_a = get_tps_parameters(landmarks_a, landmarks_b)

    # Warp the patch
    canvas = frame.copy() # frame on which it has warp
    warped_b = warp_patch(landmarks_b, warped_params_of_b, frame, canvas, rect_b)
    warped_a = warp_patch(landmarks_a, warped_params_of_a, frame, canvas, rect_a)

    if args.display:
        cv2.imshow("frame",frame)
        cv2.imshow("warped_frame",canvas)

    # Display
    if args.display:
        cv2.waitKey(0)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display images")
    args = parser.parse_args()
    main(args)
