import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from inference_engine import TensorRTInfer
from scipy.io import loadmat
import copy

def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret

def colorEncode(labelmap, colors, mode='BGR'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb

class image_converter:

 def __init__(self):
    self.bridge = CvBridge()
    # print
    rospy.Subscriber('/zedm/zed_node/left/image_rect_color', Image, self.callback1)

    # rospy.Subscriber('/tesse/left_cam/rgb/image_raw', Image,self.callback1)
    print('Loading model')
    # self.sem_engine = TensorRTInfer('/seg_workspace/hrnet_engine.trt')
    self.sem_engine = TensorRTInfer('/seg_workspace/hrnet_zedm_engine.trt')
    print('Model loaded')
    self.warmup_model()
    print('Warmup done')
    self.colors = loadmat('data/color150.mat')['colors']
    # self.colors = np.load('tesse_color_mat.npy')
    self.sem_image = None
    self.pub = rospy.Publisher('sem_image', Image, queue_size=1)

 def warmup_model(self):
    for i in range(10):
        cv_image_RGB = np.random.rand(360,640,3)
        rgb_image = np.array(cv_image_RGB)
        # print('1')
        
        # # Normalize using the provided means and stds
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        rgb_image = (rgb_image - mean) / std
        # print('12')

        rgb_image = rgb_image[np.newaxis, :, :, :].transpose(0, 3, 1, 2).astype(np.float16)
        pred = self.sem_engine.infer(rgb_image)[0][0]


 def callback1(self,imgRGB):
    global i1
    # print('0')
    cv_image_RGB = self.bridge.imgmsg_to_cv2(imgRGB, "bgr8")
    # print('01')
    rgb_image = np.array(cv_image_RGB)/255.0
    # print('1')
    
    # # Normalize using the provided means and stds
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    rgb_image = (rgb_image - mean) / std
    # print('12')

    rgb_image = rgb_image[np.newaxis, :, :, :].transpose(0, 3, 1, 2).astype(np.float16)
    pred = self.sem_engine.infer(rgb_image)[0][0]
    pred = np.int32(pred)
    pred_color = colorEncode(pred, self.colors).astype(np.uint8)

    pred_color_msg = self.bridge.cv2_to_imgmsg(pred_color, "bgr8")
    pred_color_msg.header = imgRGB.header
    # self.sem_image = pred_color_msg
    self.pub.publish(pred_color_msg)
 

 

def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
 main(sys.argv)
