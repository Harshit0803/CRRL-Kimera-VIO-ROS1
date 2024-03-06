FROM osrf/ros:noetic-desktop-full
LABEL maintainer="hs4851@nyu.edu"

# Update and install required packages
RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

# Install ROS Noetic ROS Base
RUN apt-get update && \
    apt-get install -y ros-noetic-ros-base && \
    echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"

# Install additional ROS packages and dependencies
RUN apt-get install -y \
    screen nano python3 python3-pip \
    ros-noetic-roscpp ros-noetic-std-msgs \
    ros-noetic-sensor-msgs ros-noetic-image-transport \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# Install and upgrade numpy
RUN python3 -m pip install numpy --upgrade

# Upgrade pip and install other Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install scipy cuda-python

RUN python3 -m pip install -v tensorrt

# Specific version of Eigen (replace 'your_eigen_version' with the actual version number)
RUN apt-get update && apt-get install -y libeigen3-dev

# Install GTSAM (consider building from source if a specific version is needed)
# Add steps to clone and build GTSAM from source here if necessary

WORKDIR /kimera_workspace/catkin_ws/src

# Environment variables for ROS
ENV ROS_DISTRO noetic
ENV ROS_ROOT /opt/ros/$ROS_DISTRO
ENV ROS_MASTER_URI http://localhost:11311
ENV ROS_PACKAGE_PATH /kimera_workspace/catkin_ws/src:$ROS_PACKAGE_PATH

# Source ROS setup script in entrypoint script
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Additional installations
RUN apt-get update && apt-get install -y \
    python3-rosdep python3-catkin-tools python3-vcstool cmake git

# Initialize and update rosdep
RUN [ -e /etc/ros/rosdep/sources.list.d/20-default.list ] || rosdep init && \
    apt-get install -y python3-rosdep && rosdep update

RUN apt-get install ros-noetic-image-geometry ros-noetic-pcl-ros ros-noetic-cv-bridge

RUN apt-get update

RUN apt-get install -y --no-install-recommends apt-utils

RUN apt-get install -y \
      cmake build-essential unzip pkg-config autoconf \
      libboost-all-dev \
      libjpeg-dev libpng-dev libtiff-dev \
# Use libvtk5-dev, libgtk2.0-dev in ubuntu 16.04 \
      libvtk7-dev libgtk-3-dev \
      libatlas-base-dev gfortran \
      libparmetis-dev \
      python3-wstool python3-catkin-tools libtool\
      libtbb-dev 

RUN pip3 install open3d --ignore-installed PyYAML
RUN pip3 install transforms3d

WORKDIR /kimera_workspace

RUN mkdir -p catkin_ws/src

WORKDIR /kimera_workspace/catkin_ws

RUN catkin init

RUN catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_TANGENT_PREINTEGRATION=OFF

RUN catkin config --merge-devel

RUN echo 'source /kimera_workspace/catkin_ws/devel/setup.bash' >> ~/.bashrc

WORKDIR /kimera_workspace/catkin_ws/src

RUN git clone https://github.com/Harshit0803/CRRL-Kimera-VIO-ROS1.git

RUN wstool init

RUN wstool merge CRRL-Kimera-VIO-ROS1/install/kimera_vio_ros_https.rosinstall

RUN wstool update

# RUN rosdep install --from-paths . --ignore-src -r -y

# RUN wstool update
RUN rm -rf CRRL-Kimera-VIO-ROS1

WORKDIR /kimera_workspace/catkin_ws


RUN /bin/bash -c "source $ROS_ROOT/setup.bash && catkin build"

COPY /launch/zedm_kimera_vio_ros.launch /kimera_workspace/catkin_ws/src/Kimera-VIO-ROS/launch/
COPY /launch/cloudFormation.launch /kimera_workspace/catkin_ws/src/Kimera-VIO-ROS/launch/
COPY /launch/kimera_vio_ros.launch /kimera_workspace/catkin_ws/src/Kimera-VIO-ROS/launch/
COPY params/zedm /kimera_workspace/catkin_ws/src/Kimera-VIO/params/zedm
###############

# RUN catkin build -j3 --cmake-args -DCMAKE_BUILD_TYPE=Release 
# RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
#                   source /kimera_workspace/catkin_ws/devel/setup.bash && \
#                   catkin init && \
#                   catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DGTSAM_USE_SYSTEM_EIGEN=ON && \
#                   catkin build"

# RUN source /kimera_workspace/catkin_ws/devel/setup.bash

# RUN apt-get update
WORKDIR /kimera_workspace

CMD ["/bin/bash"]



# catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release -DGTSAM_USE_SYSTEM_EIGEN=ON
