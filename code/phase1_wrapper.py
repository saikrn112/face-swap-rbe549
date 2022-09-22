import cv2
import numpy as np
from dlib import full_object
from typing import List, Tuple


def get_fiducial_landmarks(image: List[List]) -> List[Tuple]:
    # Confirm no. of points for different faces are same
    pass

def get_delaunay_triangulation(landmarks: List[Tuple]) -> List[List[int]]:
    # subdiv.getTriangleList()
    pass

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

def main():
    # Read image/s
    # Save crops of faces in image frame
    # Save image_rects (location of crop wrt the original image/frame)
    # For both crops
        # Get facial landmarks
        # Get delaunay triangulation (Check if order of triangle vertices is consistent for both)
        # Show triangulation

    # Get inverse warpings for both crops
    # Put warped crops on original image/frame
    # Display
    pass


if __name__ == '__main__':
    main()