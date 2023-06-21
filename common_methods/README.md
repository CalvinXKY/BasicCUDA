
## compile
Print information in kernel:
```
$ nvcc -lcuda print_any.cu -o print_any
```

Managed memory:
```
$ nvcc -lcuda um_demo.cu -o um_demo
```

Zero copy:
```
$ nvcc -lcuda -I../memory_opt/ zero_copy.cu -o zero_run
```

Shared memory:
```
$ nvcc -lcuda -I../memory_opt/ shared_mem.cu -o smem_run
```

Multi streams:
```
$ nvcc -lcuda streams.cu -o streamd_demo
```

## run
```
$ ./print_any
$ ./um_demo
```

## profile
### CUDA nvprof
Arch <= 7.5  e.g. Volta.
```
$ nvprof ./um_demo
```

Arch >= 8.0 e.g. Ampere:
```
$ nsys nvprof um_demo
```
### gprof

step1: compile with -pg
```
$ nvcc -pg -lcuda um_demo.cu -o um_demo
```
step2: run exe
```
$ ./um_demo
```
(will get a file: gmon.out)

step3: print info
```
$ gprof ./um_demo
```
Result e.g.:
```
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total
 time   seconds   seconds    calls  ns/call  ns/call  name
 62.50      0.03     0.03  1048576    23.84    23.84  std::fmax(float, float)
 25.00      0.04     0.01                             main
 12.50      0.04     0.01  1048576     4.77     4.77  std::fabs(float)
  0.00      0.04     0.00        2     0.00     0.00  cudaError cudaMallocManaged<float>(float**, unsigned long, unsigned int)
  0.00      0.04     0.00        2     0.00     0.00  dim3::dim3(unsigned int, unsigned int, unsigned int)
  0.00      0.04     0.00        1     0.00     0.00  _GLOBAL__sub_I_main
  0.00      0.04     0.00        1     0.00     0.00  cudaError cudaLaunchKernel<char>(char const*, dim3, dim3, void**, unsigned long, CUstream_st*)
  0.00      0.04     0.00        1     0.00     0.00  __device_stub__Z3addiPfS_(int, float*, float*)
  0.00      0.04     0.00        1     0.00     0.00  add(int, float*, float*)
  0.00      0.04     0.00        1     0.00     0.00  __static_initialization_and_destruction_0(int, int)
  0.00      0.04     0.00        1     0.00     0.00  ____nv_dummy_param_ref(void*)
  0.00      0.04     0.00        1     0.00     0.00  __sti____cudaRegisterAll()
  0.00      0.04     0.00        1     0.00     0.00  __nv_cudaEntityRegisterCallback(void**)
  0.00      0.04     0.00        1     0.00     0.00  __nv_save_fatbinhandle_for_managed_rt(void**)

 %         the percentage of the total running time of the
time       program used by this function.

cumulative a running sum of the number of seconds accounted
 seconds   for by this function and those listed above it.

 self      the number of seconds accounted for by this
seconds    function alone.  This is the major sort for this
           listing.

calls      the number of times this function was invoked, if
           this function is profiled, else blank.

 self      the average number of milliseconds spent in this
ms/call    function per call, if this function is profiled,
           else blank.

 total     the average number of milliseconds spent in this
ms/call    function and its descendents per call, if this
           function is profiled, else blank.

name       the name of the function.  This is the minor sort
           for this listing. The index shows the location of
           the function in the gprof listing. If the index is
           in parenthesis it shows where it would appear in
           the gprof listing if it were to be printed.
```

