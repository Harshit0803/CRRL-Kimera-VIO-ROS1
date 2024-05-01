import numpy as np
import tensorrt as trt
from cuda import cuda, cudart
import time
import cv2
import os


def check_cuda_err(err):
    if err not in [cuda.CUresult.CUDA_SUCCESS, cudart.cudaError_t.cudaSuccess]:
        raise RuntimeError(f"Cuda Error: {err}")


def cuda_call(call):
    err, *res = call
    check_cuda_err(err)
    return res[0] if len(res) == 1 else res


def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.nbytes
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.nbytes
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


class TensorRTInfer:
    """
    Implements inference for the Model TensorRT engine.
    """

    def __init__(self, engine_path: str):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        self.logger = trt.Logger(trt.Logger.ERROR)

        trt.init_libnvinfer_plugins(self.logger, "")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.inputs, self.outputs, self.allocations = [], [], []

        for i in range(self.engine.num_io_tensors):
            # is_input = self.engine.binding_is_input(i)
            # name = self.engine.get_binding_name(i)
            # dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            # shape = self.engine.get_binding_shape(i)
            
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.engine.get_tensor_shape(name)

            
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda_call(cudart.cudaMalloc(size))
            binding = {
                'index': i, 'name': name, 'dtype': dtype, 'shape': list(shape),
                'allocation': allocation, 'size': size
            }
            if is_input:
                self.batch_size = shape[0]
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            self.allocations.append(allocation)

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        specs = []
        for i in self.inputs:
            specs.append((i['shape'], i['dtype']))
        return specs

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of
            each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch):
        """
        Execute inference on a batch of images. The images should already be batched and
        preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale
            postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        assert len(batch) == len(self.inputs)

        for i in range(len(self.inputs)):
            memcpy_host_to_device(self.inputs[i]['allocation'], np.ascontiguousarray(batch[i]))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            memcpy_device_to_host(outputs[o], self.outputs[o]['allocation'])
        return outputs

if __name__=="__main__":
    sem_engine = TensorRTInfer('hrnet_engine.trt')
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print(sem_engine.inputs)
    print(sem_engine.outputs)
    # print(sem_engine.allocations)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')



    # # rgb_image (B,C,W,H) 
    # # rgb_image = np.random.rand(376,672,3)
    rgb_image = np.array(cv2.imread('1260.png'))/255.0
    
    # # Normalize using the provided means and stds
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    rgb_image = (rgb_image - mean) / std

    rgb_image = rgb_image[np.newaxis, :, :, :].transpose(0, 3, 1, 2).astype(np.float16)
    print(rgb_image.shape)
    # # # print(rgb_image)
    sem_image = sem_engine.infer(rgb_image)[0][0]
    
    plt.imshow(sem_image, cmap='gray')
    plt.axis('off')  # to turn off the axis numbers and ticks
    plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)
    t1 = time.time()
    for i in range(100):
        sem_image = sem_engine.infer(rgb_image)[0][0]
    print("FPS:",100/(time.time()-t1))
