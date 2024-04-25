# #!/usr/bin/env python
# import rospy
# from sensor_msgs.msg import Imu
# from scipy.signal import butter, lfilter, lfilter_zi

# class ButterFilter:
#     def __init__(self, order=2, cutoff=10, fs=400):
#         nyq = 0.5 * fs
#         normal_cutoff = cutoff / nyq
#         self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
#         # self.zi = [0] * max(len(self.a), len(self.b))
#         self.zi = lfilter_zi(self.b, self.a) * 0  # Correctly initialize zi based on filter coefficients

#     def apply(self, data):
#         y, self.zi = lfilter(self.b, self.a, data, zi=self.zi)
#         return y[0]

# class IMUFilterNode:
#     def __init__(self):
#         rospy.init_node('imu_filter_node', anonymous=True)
#         self.imu_pub = rospy.Publisher('output_imu', Imu, queue_size=100)
#         self.imu_sub = rospy.Subscriber('/zedm/zed_node/imu/data', Imu, self.imu_callback)
        
#         # Filters for each IMU data component
#         self.butter_ax = ButterFilter()
#         self.butter_ay = ButterFilter()
#         self.butter_az = ButterFilter()
#         self.butter_wx = ButterFilter()
#         self.butter_wy = ButterFilter()
#         self.butter_wz = ButterFilter()

#         self.list = []

#     def imu_callback(self,msg):
#         filtered_imu = Imu()
#         filtered_imu.header = msg.header
#         self.list.append(linear_acceleration.x)
#         if len(self.list)>50:
#             self.list = self.list[1:]
#             filtered_imu.linear_acceleration.x = self.butter_ax.apply(self.list)[-1]

#         else:
#             self.imu_pub.publish(msg)


#     def imu_callback(self, msg):
#         filtered_imu = Imu()
#         filtered_imu.header = msg.header
#         filtered_imu.linear_acceleration.x = self.butter_ax.apply(msg.linear_acceleration.x)
#         filtered_imu.linear_acceleration.y = self.butter_ay.apply(msg.linear_acceleration.y)
#         filtered_imu.linear_acceleration.z = self.butter_az.apply(msg.linear_acceleration.z)
#         filtered_imu.angular_velocity.x = self.butter_wx.apply(msg.angular_velocity.x)
#         filtered_imu.angular_velocity.y = self.butter_wy.apply(msg.angular_velocity.y)
#         filtered_imu.angular_velocity.z = self.butter_wz.apply(msg.angular_velocity.z)
        
#         self.imu_pub.publish(filtered_imu)
        
# if __name__ == '__main__':
#     try:
#         node = IMUFilterNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass


#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from scipy.signal import butter, lfilter

class ButterFilter:
    def __init__(self, order=2, cutoff=15, fs=400):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)

    def apply(self, data):
        y = lfilter(self.b, self.a, data)
        return y

class IMUFilterNode:
    def __init__(self):
        rospy.init_node('imu_filter_node', anonymous=True)
        self.imu_pub = rospy.Publisher('output_imu', Imu, queue_size=200)
        self.imu_sub = rospy.Subscriber('/zedm/zed_node/imu/data', Imu, self.imu_callback)
        
        # Filters for each IMU data component
        self.butter = ButterFilter()
        self.x_accel_list = []
        self.y_accel_list = []
        self.z_accel_list = []
        self.x_linear_list = []
        self.y_linear_list = []
        self.z_linear_list = []

    def imu_callback(self, msg):
        self.x_accel_list.append(msg.angular_velocity.x)
        self.y_accel_list.append(msg.angular_velocity.y)
        self.z_accel_list.append(msg.angular_velocity.z)
        self.x_linear_list.append(msg.linear_acceleration.x)
        self.y_linear_list.append(msg.linear_acceleration.y)
        self.z_linear_list.append(msg.linear_acceleration.z)
        if len(self.x_accel_list) >= 100:
            filtered_x_accel = self.butter.apply(self.x_accel_list)
            filtered_y_accel = self.butter.apply(self.y_accel_list)
            filtered_z_accel = self.butter.apply(self.z_accel_list)
            filtered_x_linear = self.butter.apply(self.x_linear_list)
            filtered_y_linear = self.butter.apply(self.y_linear_list)
            filtered_z_linear = self.butter.apply(self.z_linear_list)
            filtered_imu = Imu()
            filtered_imu.header = msg.header
            
            # Use the last filtered value
            filtered_imu.angular_velocity.x = filtered_x_accel[-1]
            filtered_imu.angular_velocity.y = filtered_y_accel[-1]
            filtered_imu.angular_velocity.z = filtered_z_accel[-1]
            filtered_imu.linear_acceleration.x = filtered_x_linear[-1]
            filtered_imu.linear_acceleration.y = filtered_y_linear[-1]
            filtered_imu.linear_acceleration.z = filtered_z_linear[-1]

            self.imu_pub.publish(filtered_imu)
            self.x_accel_list.pop(0)  # Remove the oldest entry to maintain list size
            self.y_accel_list.pop(0)
            self.z_accel_list.pop(0)
            self.x_linear_list.pop(0)
            self.y_linear_list.pop(0)
            self.z_linear_list.pop(0)
        else:
            # Publish the original message if not enough data has been collected
            self.imu_pub.publish(msg)

if __name__ == '__main__':
    try:
        node = IMUFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
