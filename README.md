# FasterDL
A collection of GPU-accelerated deep learning algorithms using cuBLAS & cuDNN.

Follow my progress on this library on my blog [here]().

---

To setup Docker image and run container instance:
```
~$ sudo docker build -t <IMAGE-NAME>:<TAG> .
~$ sudo docker run -it --rm --gpus all -v.:/<VOLUME-NAME> <IMAGE-NAME>
```
or if you're using VSCode, use the Dev Container extension pack and open the root in container *(recommended)*.

---

Please note that you must have...
- NVIDIA CUDA-compatible GPU with compute capability of at least 3.0
- Proper NVIDIA drivers installed (run `nvidia-smi` to confirm)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.1/index.html)

---
