# Memory Operations 
GPU memory architecture is similar with CPU's, however it has some features 
to satisfy with parallel threads. This section dedicated to help you know parts of these features. 

## Build and Run
### Build all:
```
$ cd <dir>
$ make
```
For different SMS, set parameter SMS. 

*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`. A100 set SMS=80
```
$ make SMS="80"
```

### Build part of them:

> Host and Device demo:
```
$ nvcc -lcuda hostAndDeviceTrans.cu -o testHost2Device
```
> Device to device demo:
```
$ nvcc -lcuda device2Device.cu -o testDevice2Device
```
> Zero copy demo:
```
$ nvcc -lcuda zeroCopy.cu -o testZeroCopy
```

> Shared memory demo:
```
$ nvcc -lcuda sharedMemory.cu -o testSharedMemory
```
### Run
Run them all (it will build all exe files if there are not):
```
sh run.sh
```

### Host and device data trans.
Run with default params:
```bash
$ ./testHost2Device 
```
Get help info:
```bash
./testHost2Device help
[Host and Device Memory Opt Demo:] - Starting...
Usage -device=n (n >= 0 for deviceID)
      -size=The size of memory for testing in bytes. Default: 20*1024*1024)
      -iter=n Iteration numbers of trans. Default:100
```

### Device to device.

Run with specify parameters, e.g:
```bash
./testDevice2Device -deviceA=0 -deviceB=3
```
Set GPU0 and GPU3 for this test. You could set the iteration and transfer data size. Use "help" for detail
```bash
./testDevice2Device help
[Device to Device Memory Opt Demo:] - Starting...
Usage -deviceA=n (n >= 0 for deviceID A. Default:0)
      -deviceB=n (n >= 0 for deviceID B. Default:1)
      -size=The size of memory for testing in bytes. Default: 20*1024*1024)
      -iter=n Iteration numbers of trans. Default:100

```
### Zero copy

In some scenarios, data used only once, called stream data, for calculation. It has an efficient way by using zero copy which could transfer data from system memory to GPU bypass Global memory. 

In this case, select "vector add" to show the advantage of the zero copy.

```
./testZeroCopy
```
Note: When increase iteration number, the throughput might be different.
The reason is that CUDA data transfer API costs a lot of time.

Compare:

./testZeroCopy -iter=1

VS.

./testZeroCopy -iter=100

### Shared memory

Shared memory could increase the speed of calculation in some scenarios. In this case, array sum is used.

Run:
```bash
./testSharedMemory
```
V100 result example:
```bash
[Shared Memory Application: Array Sum.] - Starting...
Sum array with shared memory.       Elapsed time: 0.006799 ms
Sum array without shared memory.    Elapsed time: 0.011756 ms

```
Note: The size of elements of arrary could affects the result. Try it as free.