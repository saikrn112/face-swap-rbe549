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

def homogenize_coords(coords: List[Tuple]) -> List[Tuple]:
    """
    nx2 -> nx3
    """
    ret = np.concatenate((coords,np.ones((coords.shape[0],1))),axis=1)
    return ret

def get_k_matrix(landmarks: List[Tuple],coords: List[Tuple]) -> List[List]:
    """
    landmarks - nx2
    coords - mx2
    output k matrix - mxn
    """
    k = np.zeros(shape=(len(coords), len(landmarks)))
    coords_x, landmarks_x = np.meshgrid(landmarks[:,0],coords[:,0])
    coords_y, landmarks_y = np.meshgrid(landmarks[:,1],coords[:,1])

    norm = abs(landmarks_y - coords_y) + abs(landmarks_x - coords_x)
    norm_sq = norm**2
    K = norm_sq*np.log(norm_sq + 1e-6)
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
               patch_rect_to  : Any):
    """
    landmarks_from  - nx2
    parameters      - (n+3)x2
    frame           - hxw
    patch_rect_to   - 4x2
    """

    x_values = np.arange(patch_rect_to.left(), patch_rect_to.right()+1, 1)
    y_values = np.arange(patch_rect_to.top(), patch_rect_to.bottom()+1, 1)

    x, y = np.meshgrid(x_values,y_values)
    coords_from = np.array([x.flatten(),y.flatten()]).T
    #coords_from = landmarks_from

    tps_mat = get_tps_matrix(landmarks_from,coords_from, True) # mxn
    print(f"tps_mat:{tps_mat.shape}")
    print(f"parameters_shape:{parameters.shape}")
    #print(f"parameters:{parameters}")

    coords_to = tps_mat @ parameters
    coords_to = coords_to.astype(int)

    print(f"coords_from:{coords_from}")
    print(f"coords_to:{coords_to}")
    """
    problems
    1: coords_to maynot be an integer
        bilinear interploation? -- scipy.interpolate.interp2d

    2: patch shapes are not same, so coords from B might not map within image A
        should we add boundaries as control points - TODO
    """

    warped_patch_shape = dlib_rect_to_shape(patch_rect_to)
    warped_patch = np.zeros(shape=(warped_patch_shape[0],warped_patch_shape[1],3))
    #print(f"landmarks_from:{landmarks_from}")
    print(f"warped_patch_shape:{warped_patch.shape}")

    frame_copy = frame.copy()
    frame_copy[coords_from[:,1],coords_from[:,0],:] = frame[coords_to[:,1],coords_to[:,0],:]

    cv2.imshow("warped_frame",frame_copy)

    return frame_copy
    
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
    frame_gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    # each frame has two faces that needs to be swapped
    rects = detector(frame, 0)
    rect_a = rects[0]
    rect_b = rects[1]

    # Get facial landmarks
    landmarks_a = get_fiducial_landmarks(predictor,frame,rect_a,args,"a")
    landmarks_b = get_fiducial_landmarks(predictor,frame,rect_b,args,"b")

    # appending rectangle boundaries as landmarks
    """
    can add rectangle's middle points for better warping?
    """
    rect_bb_coords_a = dlib_rect_to_bbox_coords(rect_a)
    rect_bb_coords_b = dlib_rect_to_bbox_coords(rect_b)
    landmarks_a = np.concatenate((landmarks_a,rect_bb_coords_a),axis=0)
    landmarks_b = np.concatenate((landmarks_b,rect_bb_coords_b),axis=0)

    # Get tps parameters for image
    warped_params_of_b = get_tps_parameters(landmarks_b, landmarks_a)

    # Warp the patch
    print(f"landmarks_to:{landmarks_a}")
    warped_b = warp_patch(landmarks_b, warped_params_of_b, frame, rect_a)

    # Get (x, y) for a
    # Construct img using (x, y)
    # Get inverse warpings for both crops
    # Put warped crops on original image/frame
    # Display
    if args.display:
        cv2.waitKey(0)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display images")
    args = parser.parse_args()
    main(args)
