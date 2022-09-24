import cv2
import numpy as np
import dlib 
from typing import List, Tuple
from utils import *
import argparse

def get_fiducial_landmarks(predictor,image: List[List], rect: List[Tuple], display=False) -> List[Tuple]:
    # get Fiducials
    fiducials_objects = predictor(image, rect)

    fiducials = shape_to_np(fiducials_objects)

    if display:
        draw_fiducials(fiducials, image)
    return fiducials

def get_delaunay_triangulation(rect: List[Tuple],landmarks: List[Tuple],display=False) -> List[List[int]]:
    landmarks_subdiv = cv2.Subdiv2D(rect);
    landmarks_subdiv.insert(landmarks.astype(int).tolist())
    triangles_list = landmarks_subdiv.getTriangleList()

    return triangles_list

def show_triangulation(triangulation: List[List[int]]) -> None:
    # draw triangles over crops of faces
    pass

def get_barycentric_matrix(tri_coords: List[int]) -> List[List]:
    pass

def check_point_within_triangle(point: Tuple, triangle: List[int]) -> bool:
    pass

def get_image_inverse_warping(
    src_img: List[List], dst_shape: Tuple[int], src_triangles:  List[List[int]], dst_triangles: List[List[int]]
) -> List[List]:

    # Get barycentric_matrix for src_triangles
    # Get inverse of barycentric_matrix for dst_triangles

    # Get a grid (np.meshgrid)
    # for point in grid:
        # for j, triangle in enumerate(dest_triangles):
        #     if point in triangle:
        #         Calculate the barycentric coordinate using barycentric_matrix_inverse
        #         Get loc in src from barycentric_matrix and barycentric coordinate
        #         homogenize loc -> (x, y)
        #           warped_src[point]= src_img[x, y]
    return warped_src

def main(args):
    display = args.display
    # Read image/s
    jenny_color = cv2.imread("../data/jenny.jpg")
    jenny_gray = cv2.cvtColor(jenny_color, cv2.COLOR_BGR2GRAY)
    chrissy_color = cv2.imread("../data/chrissy.jpg")
    chrissy_gray = cv2.cvtColor(chrissy_color, cv2.COLOR_BGR2GRAY)
    predictor_model = "../data/shape_predictor_68_face_landmarks.dat"

    # Initialize frontal face detector and shape predictor:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)

    # assuming 2 different images 
    # each image has single face that needs to be swapped
    rect_jenny = detector(jenny_color , 0)[0]
    rect_chrissy = detector(chrissy_color , 0)[0]

    # Get facial landmarks
    jenny_fiducials = get_fiducial_landmarks(predictor,jenny_color,rect_jenny)
    chrissy_fiducials = get_fiducial_landmarks(predictor,chrissy_color,rect_chrissy)

    # Get delaunay triangulation (Check if order of triangle vertices is consistent for both)
    rect_jenny_tuple = dlib_fiducials_to_tuple(rect_jenny)
    rect_chrissy_tuple = dlib_fiducials_to_tuple(rect_chrissy)
    jenny_triangle_list = get_delaunay_triangulation(rect_jenny_tuple,jenny_fiducials)
    chrissy_triangle_list = get_delaunay_triangulation(rect_chrissy_tuple, chrissy_fiducials)

    # Show triangulation
    if display:
        draw_delaunay(jenny_color,jenny_triangle_list,(255,255,255))
        draw_delaunay(chrissy_color,chrissy_triangle_list,(255,255,255))


    # Get inverse warpings for both crops
    # Put warped crops on original image/frame
    # Display
    pass


if __name__ == '__main__':
    main()

    parser = argparse.ArgumentParser()
    parser.add_argument('--display',default=False,help="to display images")
    args = parser.parse_args()
    main(args)
