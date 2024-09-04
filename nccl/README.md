# NCCL C++ Examples



## Compile

Clone this git lib to your local env, such as /home/xky/

Requirements:
* CUDA
* NVIDIA NCCL (optimized for NVLink)
* Open-MPI (option)

Recommend using docker images：

```shell
docker pull nvcr.io/nvidia/pytorch:24.07-py3
```

If there is docker-ce, run docker:
```shell
sudo docker run  --net=host --gpus=all -it -e UID=root --ipc host --shm-size="32g" \
-v /home/xky/:/home/xky \
-u 0 \
--name=nccl2 nvcr.io/nvidia/pytorch:24.07-py3 bash
```
Others:
```shell
docker run \
  --runtime=nvidia \
  --privileged \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidia1:/dev/nvidia1 \
  --device /dev/nvidia2:/dev/nvidia2 \
  --device /dev/nvidia3:/dev/nvidia3 \
  --device /dev/nvidia4:/dev/nvidia4 \
  --device /dev/nvidia5:/dev/nvidia5 \
  --device /dev/nvidia6:/dev/nvidia6 \
  --device /dev/nvidia7:/dev/nvidia7 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
  --device /dev/infiniband:/dev/infiniband \
  -v /usr/local/bin/:/usr/local/bin/ \
  -v /opt/cloud/cce/nvidia/:/usr/local/nvidia/ \
  -v /home/xky/:/home/xky \
  --ipc host \
  --net host \
  -it \
  -u root \
  --name nccl_env \
nvcr.io/nvidia/pytorch:24.07-py3 bash
```


Enter the git directory and run makefile
```shell
cd /home/xky/BasicCUDA/nccl/
make
```
If there is MPI lib in env, could compile MPI case:
```shell
make mpi
```

## Run 

### Single node

```shell
./multi_devices_per_thread
./one_devices_per_thread
./nonblocking_double_streams
```

Set DEBUG=1 would print some debug information.  
Could change ranks number by set '--nranks'. e.g:

```shell
DEBUG=1 ./nonblocking_double_streams --nranks 8
```

MPI case run:
```shell
mpirun -n 6 --allow-run-as-root ./nccl_with_mpi
```

### Multi nodes

Two nodes case: using socket connection for nccl init.

Server run in one:
```shell
./node_server
```

Client run in another one, e.g. Server IP: 10.10.1.1
```shell
./node_client --hostname 10.10.1.1
```

Add some envs:
```shell
# server:
NCCL_DEBUG=INFO NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 ./node_server --port 8066 --nranks 8
# client:
NCCL_DEBUG=INFO  NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 ./node_client --hostname 10.10.1.1 --port 8066  --nranks 8
```

