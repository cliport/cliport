"""Utility functions"""
import numpy as np
import cv2
from transforms3d._gohlketransforms import quaternion_matrix, translation_matrix, euler_matrix, quaternion_from_euler, quaternion_from_matrix

def get_origin_from_bbox(bbox):
    """Interpret origin from bounding box coordinates"""
    boxw, boxh = abs(bbox[0][0] - bbox[1][0]), abs(bbox[0][1] - bbox[1][1])
    origin = (np.min([bbox[0][0], bbox[1][0]]) + boxw//2,
                np.min([bbox[0][1], bbox[1][1]]) + boxh//2)
    return origin, boxw, boxh

def get_line_theta(bbox, cursor):
    """Get line to draw and appropriate theta value (rotation angle)"""
    origin, boxw, boxh = get_origin_from_bbox(bbox)
    # we mirror the y-axis because value of it is 0 at the top
    perpendicular = -(cursor[1] - origin[1])
    base = cursor[0] - origin[0]
    # Get theta
    theta = np.arctan2(perpendicular, base)
    # Get line
    radius = np.sqrt(boxw**2 + boxh**2)/2
    px = int(radius*np.cos(theta + np.pi)) + origin[0]
    py = int(radius*np.sin(theta + np.pi)) + origin[1]
    px2 = int(radius*np.cos(theta)) + origin[0]
    py2 = int(radius*np.sin(theta)) + origin[1]
    # We mirror the drawn line along X-Axis so that the line follows the cursor
    line = ((px, py2), (px2, py))
    theta = theta*180/np.pi
    if theta < 0:
        theta = theta + 360
    return line, theta

def depth_to_heatmap(depth) -> None:
    """Normalize depth image to color heatmap for display"""
    valid = (depth != 0)
    ranged = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) # dmin -> 0.0, dmax -> 1.0
    ranged[ranged < 0] = 0 # saturate
    ranged[ranged > 1] = 1
    output = 1.0 - ranged # 0 -> white, 1 -> black
    output[~valid] = 0 # black out invalid
    output **= 1/2.2 # most picture data is gamma-compressed
    output = cv2.normalize(output, output, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    
    return output

def draw_on_disp_img(img, data, bbox_color, rot_color, lang_goal = None) -> None:
    """Draw on display image of snapshot given pick/place data"""
    # Draw bbox
    if len(data['bbox']) == 2:
        cv2.rectangle(img, data['bbox'][0], data['bbox'][1], bbox_color, 2, 8)
        # Draw rotation line
        if len(data['rotline']) == 2:
            cv2.line(img, data['rotline'][0], data['rotline'][1], rot_color, 2)
            cv2.circle(img, data['rotline'][1], 5, rot_color, 2)
            # Draw rotation angle text
            text = f"K={data['rotation']}"
            origin, width, height = get_origin_from_bbox(data['bbox'])
            tx = origin[0] - width//2
            ty = np.max([0, origin[1] - height//2 - 10])
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2, cv2.LINE_AA)
        if lang_goal is not None:
            height, width = img.shape[:2]
            ty = height - 10
            tx = 10
            max_intensity = np.iinfo(img.dtype).max
            color = (max_intensity, max_intensity, max_intensity)
            cv2.putText(img, lang_goal, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.
  
    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.
  
    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.
  
    Args:
      points: HxWx3 float array of 3D points in camera coordinates.
      transform: 4x4 float array representing a rigid transformation matrix.
  
    Returns:
      points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                            'constant', constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points


def get_quaternion_from_yaw(yaw):
    """Get quaternion from yaw angle"""
    quaternion = quaternion_from_euler(0, 0, yaw, 'rxyz')
    return [quaternion[-1], *quaternion[:-1]]
    

def get_pose44(position, rotation):
    """Get a pose 4x4 matrix given a position and rotation (in quaternion) vectors"""
    return np.dot(translation_matrix(position), quaternion_matrix(rotation))


def get_avg_3d_centroid(depth, bbox, intrinsics, extrinsics):
    """Using depth map and bounding box in pixel coordinate, find the equivalent average 3d centroid in world coordinate system"""
    # bbox = np.sort(bbox, axis=0)
    xyz = get_pointcloud(depth, np.array(intrinsics))
    xyz = xyz[bbox[0][1]: bbox[1][1], bbox[0][0]: bbox[1][0]]
    nz_xyz = np.nonzero(xyz[:,:,2])
    centroid_camera = [np.median(xyz[:, :, 0][nz_xyz]), np.median(xyz[:, :, 1][nz_xyz]), np.median(xyz[:, :, 2][nz_xyz])]
    # Convert mm to m units
    centroid_camera = [float(x)/1000 for x in centroid_camera]
    # Construct transformation matrix
    position = translation_matrix(extrinsics['xyz'])
    ## XYZW to WXYZ
    quaternion = [extrinsics['quaternion'][-1], *extrinsics['quaternion'][:-1]]
    rotation = quaternion_matrix(quaternion)
    transform = np.dot(position, rotation)
    # Apply transformation
    centroid_world = np.dot(transform, np.array([[*centroid_camera, 1]]).T)[:-1]

    return [float(x) for x in centroid_world], centroid_camera


def get_relative_orientation(reference, yaw_rotation):
    """Get orientation relative to reference. Reference is in quaternion (WXYZ) while rotation is given in yaw radians. Returned orientation is in quaternion (WXYZ)"""
    ref_matrix = quaternion_matrix(reference)
    rel_matrix = euler_matrix(0, 0, yaw_rotation, 'sxyz')
    res_matrix = np.dot(rel_matrix, ref_matrix)
    
    return quaternion_from_matrix(res_matrix)
