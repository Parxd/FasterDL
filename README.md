# FasterDL
A collection of GPU-accelerated deep learning algorithms using cuBLAS & CUDA.

Follow my progress on this library on my blog [here]().

---

To setup Docker image and run container instance:
```docker
sudo docker build -t <IMG-NAME>:<TAG> .
sudo docker run -it --rm --gpus all -v.:/<VOLUME-NAME> <IMAGE-NAME>
```

Please note that you must have...
- NVIDIA CUDA-compatible GPU with compute capability of at least 3.0
- Proper NVIDIA drivers installed (`nvidia-smi` to confirm)
- !(NVIDIA Container Toolkit)[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.1/index.html]
