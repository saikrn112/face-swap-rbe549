import numpy as np
import cv2
import dlib
import random
from typing import List, Tuple, Any

def get_fiducial_landmarks(predictor,image: List[List], rect: List[Tuple], args, window_name) -> List[Tuple]:
    # get Fiducials
    fiducials_objects = predictor(image, rect)

    fiducials = shape_to_np(fiducials_objects)
    print(f"fiducials_shape:{fiducials.shape}")

    return fiducials

def get_exclusive_landmarks(landmarks_a, landmarks_b, args):
    exclude_landmarks = set()
    list_a = set([])
    for idx, landmark in enumerate(landmarks_a):
        if tuple(landmark.tolist()) in list_a:
            exclude_landmarks.add(idx)
            if args.debug:
                print(f"index of non-unique landmark a {idx}, {landmark}")
        list_a.add(tuple(landmark.tolist()))


    list_b = set([])
    for idx, landmark in enumerate(landmarks_b):
        if tuple(landmark.tolist()) in list_b:
            exclude_landmarks.add(idx)
            if args.debug:
                print(f"index of non-unique landmark b {idx}, {landmark}")
        list_b.add(tuple(landmark.tolist()))

    # Remove fiducials that are causing inv to be singular
    landmarks_a = np.delete(landmarks_a,list(exclude_landmarks),axis=0)
    landmarks_b = np.delete(landmarks_b,list(exclude_landmarks),axis=0)
    return landmarks_a, landmarks_b

def append_rect_coords_to_landmarks(landmarks: List[Tuple], rect: Any) -> List[Tuple]:
    """
    TBF
    """
    # appending rectangle boundaries as landmarks
    rect_bb_coords = dlib_rect_to_bbox_coords(rect)

    # appending centres of bounding rect as landmarks
    rect_side_centers = get_centers_of_rect_sides(rect)

    return np.concatenate( (landmarks, rect_bb_coords, rect_side_centers), axis=0).astype(int)


def homogenize_coords(coords: List[Tuple]) -> List[Tuple]:
    """
    nx2 -> nx3
    """
    ret = np.concatenate((coords,np.ones((coords.shape[0],1))),axis=1)
    return ret

def get_mask_and_bounding_box(frame, landmarks):
    mask = np.zeros(frame.shape, frame.dtype)
    hull_points_to = cv2.convexHull(landmarks)
    mask = cv2.fillConvexPoly(mask, hull_points_to, (255,255,255))
    box = cv2.boundingRect(np.float32([hull_points_to.squeeze()]))
    return mask,box

def blend_frame_and_patch(frame,patch,mask,box):
    x,y,w,h = box
    cx,cy = (2*x+w)//2 , (2*y+h)//2
    output = cv2.seamlessClone(np.uint16(patch),frame,mask,tuple([cx,cy]), cv2.NORMAL_CLONE)
    return output

    
# Define what landmarks you want:
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_BRIDGE_POINTS = list(range(27, 31))
LOWER_NOSE_POINTS = list(range(31, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
ALL_POINTS = list(range(0, 68))

def shape_to_np(dlib_shape, dtype="int"):
    """Converts dlib shape object to numpy array"""

    # Initialize the list of (x,y) coordinates
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)

    # Loop over all facial landmarks and convert them to a tuple with (x,y) coordinates:
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)

    # Return the list of (x,y) coordinates:
    return coordinates

def draw_shape_lines_all(np_shape, image):
    """Draws the shape using lines to connect between different parts of the face(e.g. nose, eyes, ...)"""

    draw_shape_lines_range(np_shape, image, JAWLINE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, LEFT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, NOSE_BRIDGE_POINTS)
    draw_shape_lines_range(np_shape, image, LOWER_NOSE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, LEFT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_OUTLINE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_INNER_POINTS, True)

def draw_shape_lines_range(np_shape, image, range_points, is_closed=False):
    """Draws the shape using lines to connect the different points"""

    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display, dtype=np.int32)
    cv2.polylines(image, [points], is_closed, (255, 255, 0), thickness=1, lineType=cv2.LINE_8)

def draw_shape_points_pos_range(np_shape, image, points):
    """Draws the shape using points and position for every landmark filtering by points parameter"""

    np_shape_display = np_shape[points]
    draw_shape_points_pos(np_shape_display, image)


def draw_shape_points_pos(np_shape, image):
    """Draws the shape using points and position for every landmark"""

    for idx, (x, y) in enumerate(np_shape):
        # Draw the positions for every detected landmark:
        cv2.putText(image, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))

        # Draw a point on every landmark position:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

def draw_shape_points(np_shape, image_orig):
    image = image_orig.copy()
    """Draws the shape using points for every landmark"""

    # Draw a point on every landmark position:
    for (x, y) in np_shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

def draw_fiducials(fiducials_list, image, window_name):
    for fiducials in fiducials_list:
        #Draw all lines connecting the different face parts:
        draw_shape_lines_all(fiducials, image)

        # Draw jaw line:
        #draw_shape_lines_range(fiducials, image, JAWLINE_POINTS)

        # Draw all points and their position:
        draw_shape_points_pos(fiducials, image)
        # You can also use:
        #draw_shape_points_pos_range(fiducials, image, ALL_POINTS)

        # Draw all shape points:
        draw_shape_points(fiducials, image)
        # Confirm no. of points for different faces are same

    cv2.imshow(f"{window_name}_fiducials",image)

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw delaunay triangles
def draw_delaunay(img, img_triangles, window_name_prefix) :

    size = img.shape
    r = (0, 0, size[1], size[0])

    for triangleList in img_triangles:
        random.seed(45)
        for t in triangleList :

            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))

            if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
                delaunay_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

    cv2.imshow(f"{window_name_prefix}_delaunay",img)

# Draw voronoi diagram
def draw_voronoi(img, subdiv,window_name_prefix) :

    ( facets, centers) = subdiv.getVoronoiFacetList([])
    print(f"facets:{len(facets)}")

    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (int(centers[i][0]), int(centers[i][1])), 3, (0, 0, 0), 1 , cv2.LINE_AA, 0)

    cv2.imshow(f"{window_name_prefix}_voronoi",img)

def dlib_rect_to_tuple(rect):
    return (rect.left(),rect.top(),rect.right(),rect.bottom())

def dlib_rect_to_shape(rect):
    return [rect.bottom() - rect.top(), rect.right() - rect.left()]

def dlib_rect_to_bbox_coords(rect):
    return np.array([[rect.left(),rect.top()],
            [rect.left(),rect.bottom()],
            [rect.right(),rect.bottom()],
            [rect.right(),rect.top()]])

def get_centers_of_rect_sides(rect):
    center_x = (rect.left() + rect.right())/2
    center_y = (rect.top() + rect.bottom())/2

    return np.array(
        [
            [rect.left(), center_y],
            [center_x, rect.bottom()],
            [rect.right(), center_y],
            [center_x, rect.top()]
        ]
    )

def get_colors(n):
    random.seed(45) 
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
