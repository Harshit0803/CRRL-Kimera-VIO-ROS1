## commands for pc

roslaunch kimera_vio_ros zedm_kimera_vio_ros.launch online:=true viz_type:=1  use_lcd:=true lcd_no_detection:=false  

rosbag play zedm_data_kim.bag --clock --topics /zedm/zed_node/depth/camera_info /zedm/zed_node/depth/depth_registered /zedm/zed_node/imu/data /zedm/zed_node/left/camera_info /zedm/zed_node/left/image_rect_color /zedm/zed_node/right/camera_info /zedm/zed_node/right/image_rect_color  

## for jetson  

roslaunch kimera_vio_ros zedm_kimera_vio_ros.launch online:=true viz_type:=1 use_lcd:=true lcd_no_detection:=false  

rosrun topic_tools throttle messages /semantic_pointcloud 5 /semantic_pointcloud2  

python3 visGlobalCloud.py  

rosbag play zedm_data_kim.bag --clock --topics /zedm/zed_node/depth/camera_info /zedm/zed_node/depth/depth_registered /zedm/zed_node/imu/data /zedm/zed_node/left/camera_info /zedm/zed_node/left/image_rect_color /zedm/zed_node/right/camera_info /zedm/zed_node/right/image_rect_color
