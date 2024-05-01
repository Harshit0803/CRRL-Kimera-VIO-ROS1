import open3d as o3d
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import message_filters
from sensor_msgs.msg import PointField
import transforms3d.euler as eul


from pose_graph_tools.msg import PoseGraph

from itertools import repeat
from typing import List, Tuple, Union

import time
import copy



def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h

def sparse_quantize(
    coords,
    voxel_size: Union[float, Tuple[float, ...]] = 1,
    *,
    return_index: bool = False,
    return_inverse: bool = False
    ):
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(
        ravel_hash(coords), return_index=True, return_inverse=True
    )
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs

def pointcloud2_to_o3d(cloud_msg):
    voxel_size = 0.2
    # Extract XYZ and RGB fields
    cloud_arr = np.array(list(pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z", "rgb"))))

    # Separate XYZ and RGB
    xyz = cloud_arr[:, 0:3]
    rgb = cloud_arr[:, 3]


    pc_ = np.round(xyz[:, :3] / voxel_size).astype(np.int32)
    pc_ -= pc_.min(0, keepdims=1)
    _, inds = sparse_quantize(pc_,
                                return_index=True,
                                return_inverse=False)

    xyz = xyz[inds]
    rgb = rgb[inds]

    dist = np.sqrt(xyz[:,0]**2+xyz[:,1]**2+xyz[:,2]**2)
    range_filter = np.where(dist<5)
    xyz = xyz[range_filter]
    rgb = rgb[range_filter]
    # Create an Open3D point cloud object and assign the points and colors
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(xyz)

    # print('3',np.asarray(cloud_o3d.colors))

    # cloud_o3d = cloud_o3d.voxel_down_sample(0.5)

    # print('4',np.asarray(cloud_o3d.colors))

    return cloud_o3d, rgb

def o3d_to_pointcloud2(cloud_o3d, rgb, frame_id="map"):
    # Extract points
    xyz = np.asarray(cloud_o3d.points)


    # Stack XYZ and RGB to create the point cloud array
    cloud_arr = np.hstack((xyz, np.expand_dims(rgb, axis=-1)))

    # Create a PointCloud2 message
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    # header = self.header

    # Define the fields for a PointCloud2 message with XYZ and RGB
    fields = [PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
              PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
              PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
              PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)]

    # Create the PointCloud2 message
    cloud_msg = pc2.create_cloud(header, fields, cloud_arr)

    return cloud_msg

def rotate_z(transformation_matrix, theta_degrees):
    # Convert theta from degrees to radians
    theta_radians = np.radians(theta_degrees)
    
    # Create a rotation matrix for rotation around the z-axis
    rotation_matrix = np.array([
        [np.cos(theta_radians), -np.sin(theta_radians), 0, 0],
        [np.sin(theta_radians), np.cos(theta_radians), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Multiply the original transformation matrix by the rotation matrix
    rotated_matrix = np.dot(transformation_matrix, rotation_matrix)
    
    return rotated_matrix

def rotate_y(transformation_matrix, theta_degrees):
    # Convert theta from degrees to radians
    theta_radians = np.radians(theta_degrees)
    
    # Create a rotation matrix for rotation around the y-axis
    rotation_matrix = np.array([
        [np.cos(theta_radians), 0, np.sin(theta_radians), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_radians), 0, np.cos(theta_radians), 0],
        [0, 0, 0, 1]
    ])
    
    # Multiply the original transformation matrix by the rotation matrix
    # Note: Depending on the order you want (rotate then transform or transform then rotate),
    # you might need to switch the order of multiplication.
    rotated_matrix = np.dot(transformation_matrix, rotation_matrix)
    
    return rotated_matrix

def rotate_x(transformation_matrix, theta_degrees):
    # Convert theta from degrees to radians
    theta_radians = np.radians(theta_degrees)
    
    # Create a rotation matrix for rotation around the x-axis
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_radians), -np.sin(theta_radians), 0],
        [0, np.sin(theta_radians), np.cos(theta_radians), 0],
        [0, 0, 0, 1]
    ])
    
    # Multiply the original transformation matrix by the rotation matrix
    rotated_matrix = np.dot(transformation_matrix, rotation_matrix)
    
    return rotated_matrix

def transform_point_cloud(o3d_pcd_temp, pose):
    # pose: the pose containing 'position' and 'orientation' (quaternion xyzw)

    o3d_pcd = copy.deepcopy(o3d_pcd_temp)
    # o3d_pcd = copy.deepcopy(o3d_pcd_temp)
    # Create a transformation matrix from the pose
    translation = np.array([pose.position.x, pose.position.y, pose.position.z])
    quaternion = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])  # w first
    transformation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
    euler_angles = eul.mat2euler(transformation_matrix)

    transformation_matrix = np.hstack((transformation_matrix, translation[...,None]))

    T_imu_in_world = np.vstack((transformation_matrix, [0, 0, 0, 1]))  # Add the bottom row for affine transformation
    euler_angles = eul.mat2euler(T_imu_in_world[:3,:3])

    T_camera_in_imu = np.array([[-0.01155918, -0.00865423,  0.99989574,  0.0021902],
                                    [-0.99988854, -0.0093498,  -0.01164002,  0.01519092],
                                        [0.00944956, -0.99991884, -0.00854519,  0.00153914],
                                        [0.,          0.,          0.,          1.        ]])

    # Apply the transformation
    T_camera_in_world = T_imu_in_world@T_camera_in_imu
    # o3d_pcd.transform(T_camera_in_imu)
    # o3d_pcd.transform(T_imu_in_world)
    o3d_pcd.transform(T_camera_in_world)
    
    

    return o3d_pcd

class VisualizeGlobalPcd():
    def __init__(self):
        self.data_dict = {}
        self.global_pcd = o3d.geometry.PointCloud()
        self.colors = []
        self.pre_lc = 0
        self.ongoing_lc = 0

        rospy.init_node('my_custom_listener', anonymous=True)
        self.last_time = rospy.Time.now()
        pose_graph_sub = message_filters.Subscriber("/kimera_vio_ros/pose_graph", PoseGraph)
        pointcloud_sub = message_filters.Subscriber("/semantic_pointcloud", PointCloud2)

        rospy.Subscriber("/kimera_vio_ros/pose_graph", PoseGraph, self.loop_closure)
        
        ts = message_filters.ApproximateTimeSynchronizer([pose_graph_sub, pointcloud_sub], 50, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)
        self.pub = rospy.Publisher("/global_pointcloud", PointCloud2, queue_size=50)
        
        rospy.spin()

    def do_loop_closure(self, pose_graph):
        global_pcd = o3d.geometry.PointCloud()
        for i in range(len(pose_graph.nodes)):
            id_key = pose_graph.nodes[i].key
            pose = pose_graph.nodes[i].pose
            if id_key in self.data_dict.keys():
                # print(id_key)
                o3d_pcd = self.data_dict[id_key]
                global_pcd_t = transform_point_cloud(o3d_pcd,pose)
                global_pcd += global_pcd_t
        self.global_pcd = global_pcd
    
    def loop_closure(self, pose_graph):
        if len(pose_graph.edges)==0:
            return
        if pose_graph.edges[-1].type==1:
            self.ongoing_lc = 1
            print('Doing loop closure')
            self.do_loop_closure(pose_graph)
            # self.pre_lc = 1
        global_cloud_msg = o3d_to_pointcloud2(self.global_pcd, self.colors)
        self.pub.publish(global_cloud_msg)
        self.ongoing_lc = 0
            

    def callback(self, pose_graph, pointcloud):
        if self.ongoing_lc == 1:
            return

        current_time = rospy.Time.now()
        if (current_time - self.last_time).to_sec() < 2.0:
            return  # Return early if less than one second has passed

        # Update the last_time to the current time
        self.last_time = current_time

        o3d_pcd, rgb = pointcloud2_to_o3d(pointcloud)
        self.colors.extend(rgb.tolist())
        pose = pose_graph.nodes[-1].pose
        key = pose_graph.nodes[-1].key

        self.data_dict[key] = o3d_pcd

        global_pcd_t = transform_point_cloud(o3d_pcd, pose)

        self.global_pcd += global_pcd_t
        # o3d.visualization.draw_geometries([self.global_pcd])
        print(len(self.global_pcd.points), len(self.colors))

        global_cloud_msg = o3d_to_pointcloud2(self.global_pcd, self.colors)
        self.pub.publish(global_cloud_msg)





if __name__ == '__main__':
    VisualizeGlobalPcd()
