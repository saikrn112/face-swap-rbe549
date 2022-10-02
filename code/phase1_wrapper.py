import cv2
import numpy as np
import dlib 
from typing import List, Tuple, Any
from utils import *
import argparse
from shapely.geometry import Point, Polygon
import time

def get_delaunay_triangulation(rect: List[Tuple],
                                landmarks: List[Tuple],
                                args,
                                frame,
                                window_name) -> List[List[int]]:
    landmarks_subdiv = cv2.Subdiv2D(dlib_rect_to_tuple(rect))
    landmarks_subdiv.insert(landmarks.tolist())
    triangles_list = landmarks_subdiv.getTriangleList()

    if args.debug:
        img_voronoi = frame.copy()
        draw_voronoi(img_voronoi,landmarks_subdiv,window_name)

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

def check_point_in_triangle(point):
    """
    point - 1 X 3
    """
    eps = 1e-6
    if ((0-eps <= point) & (point <= 1+eps)).all() \
            and (0-eps < np.sum(point) <= 1+eps):
        return True
    return False

def point_within_triangle(point: Tuple, dst_inv: List[List]) -> List[Any]:
    """
    point: [x, y, 1]
    dst_inv: 3x3
    """
    point = np.array([point]).T
    bary_coords = dst_inv @ point

    if ((0 <= bary_coords) & (bary_coords <= 1)).all() \
        and (0 < np.sum(bary_coords,axis=0) <= 1):
        return True, bary_coords

    return False,None

def point_within_triangle1(point: Tuple, dst_inv: List[List], polygon) -> List[Any]:

    cond = polygon.contains(Point(point[0], point[1]))
    if cond:
        point = np.array([point]).T
        bary_coords = dst_inv @ point
        return True, bary_coords

    return False, None

def construct_polygon(triangle):
    triangle = np.reshape(triangle, (3,2))
    polygon = Polygon(map(Point,triangle))
    return polygon

def get_image_inverse_warping(  landmarks_from  :  List[Tuple], 
                                frame           :  List[List], 
                                canvas          :  List[List],
                                dst_rect_dlib   :  Any,
                                src_triangles   :  List[List[int]], 
                                dst_triangles   :  List[List[int]], 
                                suffix          :  str,
                                args)           -> List[List]:

    """
    we are replacing dst intensities from src intensities
    for that we are first inverse warping the dst coords to src
    """
    # Filter coords from patch that lie within the convex hull of landmarks
    landmarks_from_raw = landmarks_from[:-8].astype(np.int32)
    mask, box = get_mask_and_bounding_box(frame,landmarks_from_raw)
    indices_from = np.argwhere(mask==(255,255,255))

    # Get barycentric_matrix for src_triangles
    src_bary_matrices = [get_barycentric_matrix(src_tri) for src_tri in src_triangles]

    # Get inverse of barycentric_matrix for dst_triangles
    dst_bary_inverses = [get_barycentric_matrix(dst_tri, True) for dst_tri in dst_triangles]
    dst_polygons = [construct_polygon(dst_tri) for dst_tri in dst_triangles]

    if args.debug:
        print(f"src_bary: {src_bary_matrices}")
        print(f"dst_inv: {dst_bary_inverses}")

    # Convert indices to coords because cv functions above changed the type
    coords_from = np.vstack((indices_from[:,1], indices_from[:,0])).T
    coords = homogenize_coords(coords_from).astype(int)

    face = np.zeros(shape=frame.shape,dtype=frame.dtype)
    for j, dst_bary_inv in enumerate(dst_bary_inverses):
        bary_coords = dst_bary_inv @ coords.T
        coords_within = np.apply_along_axis(check_point_in_triangle,axis=0,arr=bary_coords)
        within_indxs = np.argwhere(coords_within==True).flatten().tolist()
        filtered_coords = coords.T[:,within_indxs]

        # Get loc in src from barycentric_matrix and barycentric coordinate
        src_loc = src_bary_matrices[j] @ bary_coords[:,within_indxs]

        # Unhomogenize loc -> (x, y)
        x = (src_loc[0,:] / src_loc[2,:]).astype(int)
        y = (src_loc[1, :] / src_loc[2, :]).astype(int)

        face[filtered_coords[1,:], filtered_coords[0,:], :] = frame[y, x, :]

    cv2.imshow(f"face_{suffix}",face)
    cv2.imshow(f"mask_{suffix}",mask)
    output = blend_frame_and_patch(canvas, face, mask, box)
    return output

def main(args):
    display = args.display
    debug = args.debug

    out = cv2.VideoWriter('../data/result_del_tri.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (600, 338))

    predictor_model = "../data/shape_predictor_68_face_landmarks.dat"

    # Initialize frontal face detector and shape predictor:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)

    cap = cv2.VideoCapture("../data/sample_video1.gif")
    while True:
        _, frame = cap.read()
        # Read image/s


        # Each frame has two faces that need to be swapped
        rects = detector(frame, 0)
        rect_a = rects[0]
        rect_b = rects[1]

        # Get facial landmarks
        landmarks_a = get_fiducial_landmarks(predictor, frame, rect_a, args, 'a')
        landmarks_b = get_fiducial_landmarks(predictor, frame, rect_b, args, 'b')

        # if args.display:
        #     img_fiducials = frame.copy()
        #     draw_fiducials([landmarks_a,landmarks_b],img_fiducials,"ab")

        # Remove fiducials that are causing inv to be singular
        landmarks_a, landmarks_b = get_exclusive_landmarks(landmarks_a, landmarks_b,args)

        # Append rect corner and center coords as additional landmarks
        landmarks_a = append_rect_coords_to_landmarks(landmarks_a, rect_a)
        landmarks_b = append_rect_coords_to_landmarks(landmarks_b, rect_b)

        # Get delaunay triangulation
        # print(landmarks_a)
        triangle_list_a = get_delaunay_triangulation(rect_a, landmarks_a, args, frame, "a")
        _               = get_delaunay_triangulation(rect_b, landmarks_b, args, frame, "b")
        triangle_list_b = get_triangulation_for_src(triangle_list_a, landmarks_a, landmarks_b, args, frame, "b")

        # Show triangulation
        # if args.display:
        #     img_tri = frame.copy()
        #     draw_delaunay(img_tri,[triangle_list_a, triangle_list_b],"a")

        # Get inverse warpings for both crops
        canvas = frame.copy() # frame on which it has warp
        canvas = get_image_inverse_warping(landmarks_b, frame, canvas, rect_b, triangle_list_a, triangle_list_b, "b", args)
        canvas = get_image_inverse_warping(landmarks_a, frame, canvas, rect_a, triangle_list_b, triangle_list_a, "a", args)

        if args.display:
            cv2.imshow("frame",frame)
            cv2.imshow("warped_img",canvas)

        out.write(canvas)

        if args.display:
            cv2.waitKey(1)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display images")
    args = parser.parse_args()
    main(args)
