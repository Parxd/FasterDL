# FasterDL
A collection of GPU-accelerated deep learning algorithms using cuBLAS & CUDA.

Follow my progress on this library on my blog [here]().

---

To setup Docker image and run container instance:
```docker
sudo docker build -t <IMG-NAME>:<TAG> .
sudo docker run -it --rm --gpus all -v.:/<VOLUME-NAME> <IMAGE-NAME>
```
