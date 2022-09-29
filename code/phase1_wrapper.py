import cv2
import numpy as np
import dlib 
from typing import List, Tuple, Any
from utils import *
import argparse

def get_delaunay_triangulation(rect: List[Tuple],
                                landmarks: List[Tuple],
                                args,
                                image_color,
                                window_name) -> List[List[int]]:
    landmarks_subdiv = cv2.Subdiv2D(dlib_rect_to_tuple(rect))
    landmarks_subdiv.insert(landmarks.tolist())
    triangles_list = landmarks_subdiv.getTriangleList()

    # Show triangulation
    if args.display:
        draw_delaunay(image_color,triangles_list,window_name)
        if args.debug:
            draw_voronoi(image_color,landmarks_subdiv,window_name)
    return triangles_list

def get_triangulation_for_src(dst_triangulation: List[List[int]],
                              src_fiducials: List[Tuple],
                              dst_fiducials: List[Tuple],
                              args,
                              image_color,
                              window_name) -> List[List[int]]:
    """
    We are getting src_triangulation from dst_triangulation
    dst_triangulation: mx6
    src_fiducials: 68x2
    dst_fiducials: 68x2
    """
    fidu_coord_to_idx = {tuple(fidu_coord): idx for idx, fidu_coord in enumerate(src_fiducials)}

    src_triangulation = []
    for triangle in dst_triangulation:

        coords1_idx = fidu_coord_to_idx[(triangle[0], triangle[1])]
        coords2_idx = fidu_coord_to_idx[(triangle[2], triangle[3])]
        coords3_idx = fidu_coord_to_idx[(triangle[4], triangle[5])]

        src_triangulation.append(
            [
                *dst_fiducials[coords1_idx],
                *dst_fiducials[coords2_idx],
                *dst_fiducials[coords3_idx]
            ]
        )

    # Show triangulation
    if args.display:
        draw_delaunay(image_color, src_triangulation, window_name)

    return src_triangulation


def get_barycentric_matrix(tri_coords: List[int], get_inv: bool=False) -> List[List]:
    """
    tri_coords is of the form [x1, y1, x2, y2, x3, y3]
    return is a 3x3 matrix
    """
    mat = np.reshape(tri_coords, (3, 2))
    mat = np.concatenate([mat, np.ones((3, 1))], axis=1).T

    if get_inv:
        try:
            inv = np.linalg.inv(mat)
            return inv
        except:
            print("singular:", tri_coords)
            exit(1)
    return mat


def point_within_triangle(point: Tuple, dst_inv: List[List]) -> List[Any]:
    """
    point: [x, y, 1]
    dst_inv: 3x3
    """
    point = np.array([point]).T
    bary_coords = dst_inv @ point

    if ((0 <= bary_coords) & (bary_coords <= 1)).all() \
        and (0 < np.sum(bary_coords,axis=0) <= 1):
        return True,bary_coords
    return False,None

def get_image_inverse_warping( frame: List[List], 
                            dst_rect_dlib: Any,
                            src_triangles:  List[List[int]], 
                            dst_triangles: List[List[int]], 
                            args) -> List[List]:

    """
    we are replacing dst intensities from src intensities
    for that we are first inverse warping the dst coords to src
    """
    # Get barycentric_matrix for src_triangles
    src_bary_matrices = [get_barycentric_matrix(src_tri) for src_tri in src_triangles]
    if args.debug:
        print(f"src_bary: {src_bary_matrices}")

    # Get inverse of barycentric_matrix for dst_triangles
    dst_bary_inverses = [get_barycentric_matrix(dst_tri, True) for dst_tri in dst_triangles]
    if args.debug:
        print(f"dst_inv: {dst_bary_inverses}")

    # Get a grid
    start_x = dst_rect_dlib.left()
    end_x = dst_rect_dlib.right()
    start_y = dst_rect_dlib.top()
    end_y = dst_rect_dlib.bottom()

    xs = np.arange(start_x, end_x+1, 1).astype(int)
    ys = np.arange(start_y, end_y+1, 1).astype(int)

    grid_xs, grid_ys = np.meshgrid(xs, ys)
    grid_xs = np.array([grid_xs.flatten()])
    grid_ys = np.array([grid_ys.flatten()])

    coords = np.concatenate((grid_xs, grid_ys), axis=0).T

    frame_copy = frame.copy()
#    frame_copy[coords[:,1],coords[:,0]] = (255,255,255)
#    cv2.imshow("white",frame_copy)

    coords = homogenize_coords(coords)

    for pt in coords:
        for j, dst_bary_inv in enumerate(dst_bary_inverses):
            in_triangle, barycentric_coord = point_within_triangle(pt, dst_bary_inv)
            if in_triangle:
                # Get loc in src from barycentric_matrix and barycentric coordinate
                src_loc = src_bary_matrices[j] @ barycentric_coord
                
                # Unhomogenize loc -> (x, y)
                x, y = int(src_loc[0]/src_loc[2]),int(src_loc[1]/src_loc[2])

                frame_copy[y, x, :] = frame[x, y, :]
    return frame_copy 

def main(args):
    display = args.display
    debug = args.debug
    predictor_model = "../data/shape_predictor_68_face_landmarks.dat"

    # Initialize frontal face detector and shape predictor:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)

    # Read image/s
    frame = cv2.imread("../data/selfie_2.jpeg")
    frame = cv2.resize(frame,(550,400))

    # Each frame has two faces that need to be swapped
    rects = detector(frame, 0)
    rect_a = rects[0]
    rect_b = rects[1]

    # Get facial landmarks
    landmarks_a = get_fiducial_landmarks(predictor, frame, rect_a, args, 'a')

    # Remove fiducials that are causing inv to be singular
    exclude_landmarks = set()
    list_a = set([])
    for idx, landmark in enumerate(landmarks_a):
        if tuple(landmark.tolist()) in list_a:
            exclude_landmarks.add(idx)
            print(f"index of non-unique landmark a {idx}, {landmark}")
        list_a.add(tuple(landmark.tolist()))

    landmarks_b = get_fiducial_landmarks(predictor, frame, rect_b, args, 'b')

    list_b = set([])
    for idx, landmark in enumerate(landmarks_b):
        if tuple(landmark.tolist()) in list_b:
            exclude_landmarks.add(idx)
            print(f"index of non-unique landmark b {idx}, {landmark}")
        list_b.add(tuple(landmark.tolist()))

    # Remove fiducials that are causing inv to be singular
    landmarks_a = np.delete(landmarks_a,list(exclude_landmarks),axis=0)
    landmarks_b = np.delete(landmarks_b,list(exclude_landmarks),axis=0)
    print(landmarks_a.shape)
    print(landmarks_b.shape)

    # Get delaunay triangulation
    triangle_list_a = get_delaunay_triangulation(rect_a, landmarks_a, args, frame, "a")
    triangle_list_b = get_triangulation_for_src(triangle_list_a, landmarks_a, landmarks_b, args, frame, "b")

    # Get inverse warpings for both crops
    warp_img = get_image_inverse_warping(frame, rect_b, triangle_list_a, triangle_list_b, args)
    
    if args.display:
        cv2.imshow("warp_img",warp_img)

    if args.display:
        cv2.waitKey(0)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display images")
    args = parser.parse_args()
    main(args)
