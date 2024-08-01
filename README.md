# Udemy_Py_TensorRT

ONNX: ONNX (Open Neural Network Exchange) is an open format built to represent machine learning models. It enables models to be transferred between different frameworks, which can facilitate deployment in various environments. Allows models to be trained in one framework (like PyTorch) and then transferred to another framework (like TensorFlow or Caffe2) for inference.
Fundamentals of ONNX: batch size(20)

ONNX is designed for model interoperability and runtime and not for training.
To visualise ONNX framework use netron.app

TENSORRT: high-performance deep learning inference library developed by NVIDIA. It is specifically designed to optimize and deploy neural network models on NVIDIA GPUs. Provides various optimizations such as layer fusion, precision calibration (FP16 and INT8), and kernel auto-tuning to maximize inference performance on NVIDIA GPUs.
36 times faster than ONNX
Always allocate 70% of GPU memory for tensorRT
TensorRT architecture can be converted to fp16,int8


Triton Server: open-source inference serving software designed to simplify the deployment of AI models at scale. It provides a flexible and efficient solution for running inference on models from various frameworks, supporting multiple backends, and optimizing for different hardware platforms.

Docker: https://github.com/Leanse1/Docker/tree/main/docker
install docker in VM: sudo apt install docker.io
                      sudo usermod -aG docker $USER
                      sudo groupadd docker
                      sudo gpasswd -a $USER docker
                      sudo systemctl restart docker
                      sudo systemctl status docker

install docker cuda toolkit: chap 22,23,24,25
install docker tensorflow: chap 26,27,28
install git, yolo, tensorrt chap 29,30

Nvidia Driver: To have proper connection b/w nvidia graphics card(rtx) and windows os.
to install nvidia driver on VM: https://www.youtube.com/watch?v=pmGfi1ldBqc

CUDA:  CUDA library is specialized for GPU-based parallelism and large-scale data operations, 
    similar to multithreading which is suited for CPU-based concurrent execution and managing tasks within the same memory space.

    workflow: pytorch call C++; C++ call cuda;
    C++ acts as bridge b/w both

Hardware(Cpu,gpu,processor) >> Kernel (Linux, windows, ubuntu) >>> User (program languages, Libraries, cuda toolkit) 
