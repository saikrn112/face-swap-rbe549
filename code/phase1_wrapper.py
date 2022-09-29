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
    landmarks_subdiv = cv2.Subdiv2D(rect)
    landmarks_subdiv.insert(landmarks.astype(int).tolist())
    triangles_list = landmarks_subdiv.getTriangleList()

    # Show triangulation
    if args.display:
        draw_delaunay(image_color,triangles_list,window_name)
        if args.debug:
            draw_voronoi(image_color,landmarks_subdiv,window_name)
    return triangles_list

def get_triangulation_for_src(dst_triangulation: List[List[int]], src_fiducials: List[Tuple], dst_fiducials, 
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
                *dst_fiducials[coords1_idx], *dst_fiducials[coords2_idx], *dst_fiducials[coords3_idx]
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
        return np.linalg.inv(mat)
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
        print(bary_coords)
        return True,bary_coords
    return False,None

def get_image_inverse_warping( src_img: List[List], 
                            dst_rect_dlib: Any,
                            src_triangles:  List[List[int]], 
                            dst_triangles: List[List[int]], 
                            args) -> List[List]:

    # Get barycentric_matrix for src_triangles
    src_bary_matrices = [get_barycentric_matrix(src_tri) for src_tri in src_triangles]
    if args.debug:
        print(f"src_bary: {src_bary_matrices}")

    # Get inverse of barycentric_matrix for dst_triangles
    dst_bary_inverses = [get_barycentric_matrix(dst_tri, True) for dst_tri in dst_triangles]
    if args.debug:
        print(f"dst_inv: {dst_bary_inverses}")


    # Get a grid
    dst_shape = dlib_rect_to_img_shape(dst_rect_dlib)
    warped_src = np.zeros((dst_shape[0],dst_shape[1],3))

    start_x = dst_rect_dlib.left()
    end_x = dst_rect_dlib.right()
    start_y = dst_rect_dlib.top()
    end_y = dst_rect_dlib.bottom()

    xs = np.arange(start_x, end_x, 1).astype(int)
    ys = np.arange(start_y, end_y, 1).astype(int)

    grid_xs, grid_ys = np.meshgrid(xs, ys)
    grid_xs = np.array([grid_xs.flatten()])
    grid_ys = np.array([grid_ys.flatten()])

    coords = np.concatenate((grid_xs, grid_ys, np.ones((1,dst_shape[0]*dst_shape[1]))), axis=0).T

    for pt in coords:
        for j, dst_bary_inv in enumerate(dst_bary_inverses):
            in_triangle, barycentric_coord = point_within_triangle(pt, dst_bary_inv)
            if in_triangle:
                print(barycentric_coord)
                # Get loc in src from barycentric_matrix and barycentric coordinate
                src_loc = src_bary_matrices[j] @ barycentric_coord

                
                # Unhomogenize loc -> (x, y)
                x, y = int(src_loc[0]/src_loc[2]),int(src_loc[1]/src_loc[2])
                print(f"x:{x} y:{y}")

                warped_src[y, x, :] = src_img[x, y, :]
    return warped_src

def main(args):
    display = args.display
    debug = args.debug
    # Read image/s
    jenny_color = cv2.imread("../data/selfie_4.jpeg")
    print(jenny_color.shape)
    jenny_color = cv2.resize(jenny_color,(550,400))
    print(jenny_color.shape)
    jenny_gray = cv2.cvtColor(jenny_color, cv2.COLOR_BGR2GRAY)
    #chrissy_color = cv2.imread("../data/chrissy.jpg")
    #chrissy_gray = cv2.cvtColor(chrissy_color, cv2.COLOR_BGR2GRAY)
    predictor_model = "../data/shape_predictor_68_face_landmarks.dat"

    # Initialize frontal face detector and shape predictor:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)

    # assuming 2 different images 
    # each image has single face that needs to be swapped
    rects = detector(jenny_color, 0)
    rect_jenny = rects[0]
    rect_chrissy = rects[1]

    rect_jenny_tuple = dlib_fiducials_to_tuple(rect_jenny)
    rect_chrissy_tuple = dlib_fiducials_to_tuple(rect_chrissy)

    # Get facial landmarks
    jenny_fiducials = get_fiducial_landmarks(predictor,jenny_color,rect_jenny,args,"ramana")
    chrissy_fiducials = get_fiducial_landmarks(predictor,jenny_color,rect_chrissy,args,"radha")

    # Get delaunay triangulation (TODO Check if order of triangle vertices is consistent for both)
    jenny_triangle_list = get_delaunay_triangulation(rect_jenny_tuple,jenny_fiducials,args,jenny_color,"ramana")
    chrissy_triangle_list = get_triangulation_for_src(jenny_triangle_list, jenny_fiducials, chrissy_fiducials, args, jenny_color, "radha")
    # chrissy_triangle_list = get_delaunay_triangulation(rect_chrissy_tuple, chrissy_fiducials,args,jenny_color,"radha")

    if args.display:
        cv2.waitKey(0)
    
    # Get inverse warpings for both crops
    get_image_inverse_warping(jenny_color, rect_jenny, jenny_triangle_list, chrissy_triangle_list, args)
    
    # Put warped crops on original image/frame
    # Display


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display',default=False,help="to display images")
    parser.add_argument('--debug',default=False,help="to display images")
    args = parser.parse_args()
    main(args)
