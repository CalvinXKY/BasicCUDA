/**
 *  The original code: https://github.com/pytorch/pytorch/blob/v1.13.1/c10/cuda/CUDACachingAllocator.h
 *  Modified for independent running.
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 * */
#ifndef THC_DEVICE_ALLOCATOR_INC
#define THC_DEVICE_ALLOCATOR_INC

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cstdio>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <vector>

using namespace std;
// original code: using stream_set = ska::flat_hash_set<cuda::CUDAStream>;
// flat_hash_set -> set:
using stream_set = set<cudaStream_t>;

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)
#endif

template <typename T> void toStringStream(std::stringstream &ss, T value)
{
    ss << value;
}

template <typename T, typename... Args> void toStringStream(std::stringstream &ss, T first, Args... args)
{
    ss << first;
    toStringStream(ss, args...);
}

template <typename... Args> std::string concatenate(Args... args)
{
    std::stringstream ss;
    toStringStream(ss, args...);
    return ss.str();
}

#define C10_CUDA_CHECK(EXPR)                                                              \
    do {                                                                                  \
        cudaError_t __err = EXPR;                                                         \
        if (__err != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA ERROR: (error code %s)!\n", cudaGetErrorString(__err)); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

// Simplified torch_check：
#define TORCH_CHECK(cond, ...)                  \
    if (!(cond)) {                              \
        printf("error info:%s", ##__VA_ARGS__); \
        exit(EXIT_FAILURE);                     \
    }

#define TORCH_INTERNAL_ASSERT(...) TORCH_CHECK(__VA_ARGS__)

#define TORCH_CHECK_WITH(cond, ...)               \
    if (!(cond)) {                                \
        cout << concatenate(__VA_ARGS__) << endl; \
        exit(EXIT_FAILURE);                       \
    }

constexpr size_t kMinBlockSize = 512;       // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;      // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;    // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;   // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;     // round up large allocations to 2 MiB

struct Stat {
    int64_t current = 0;
    int64_t peak = 0;
    int64_t allocated = 0;
    int64_t freed = 0;
};

enum struct StatType : uint64_t {
    AGGREGATE = 0,
    SMALL_POOL = 1,
    LARGE_POOL = 2,
    NUM_TYPES = 3 // remember to update this whenever a new stat type is added
};

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
    // COUNT: allocations requested by client code
    StatArray allocation;
    // COUNT: number of allocated segments from cudaMalloc().
    StatArray segment;
    // COUNT: number of active memory blocks (allocated or used by stream)
    StatArray active;
    // COUNT: number of inactive, split memory blocks (unallocated but can't be
    // released via cudaFree)
    StatArray inactive_split;

    // SUM: bytes requested by client code
    StatArray allocated_bytes;
    // SUM: bytes reserved by this memory allocator (both free and used)
    StatArray reserved_bytes;
    // SUM: bytes within active memory blocks
    StatArray active_bytes;
    // SUM: bytes within inactive, split memory blocks
    StatArray inactive_split_bytes;

    // COUNT: total number of failed calls to CUDA malloc necessitating cache
    // flushes.
    int64_t num_alloc_retries = 0;

    // COUNT: total number of OOMs (i.e. failed calls to CUDA after cache flush)
    int64_t num_ooms = 0;

    // COUNT: total number of oversize blocks allocated from pool
    Stat oversize_allocations;

    // COUNT: total number of oversize blocks requiring malloc
    Stat oversize_segments;

    // SIZE: maximum block size that is allowed to be split.
    int64_t max_split_size = 0;
};

struct Context {
    virtual ~Context()
    {
    }
};

typedef std::unique_ptr<Context> (*CreateContextFn)(void);

struct History {
    void *addr;
    size_t real_size;                 // unrounded, actually requested size
    std::unique_ptr<Context> context; // per-watcher context
    std::unique_ptr<History> next;    // when blocks are merged we keep records of
                                      // what used to be in the block
};

// Struct containing info of an allocation block (i.e. a fractional part of a
// cudaMalloc)..
struct BlockInfo {
    int64_t size = 0;
    int32_t gc_counter = 0;
    bool allocated = false;
    bool active = false;
    History *history = nullptr; // borrowed reference because it is owned by the allocator
};

// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
struct SegmentInfo {
    int64_t device = 0;
    int64_t address = 0;
    int64_t total_size = 0;
    int64_t allocated_size = 0;
    int64_t active_size = 0;
    cudaStream_t stream = 0;
    bool is_large = false;
    std::vector<BlockInfo> blocks;
};

struct Block;
struct PrivatePool; // CUDA graphs helper
typedef bool (*Comparison)(const Block *, const Block *);

struct BlockPool {
    BlockPool(Comparison comparator, bool small, PrivatePool *private_pool = nullptr)
        : blocks(comparator)
        , is_small(small)
        , owner_PrivatePool(private_pool)
    {
    }
    std::set<Block *, Comparison> blocks;
    const bool is_small;
    PrivatePool *owner_PrivatePool;
};

struct Block {
    int device;             // gpu
    cudaStream_t stream;    // allocation stream
    stream_set stream_uses; // streams on which the block was used
    size_t size;            // block size in bytes
    BlockPool *pool;        // owning memory pool
    void *ptr;              // memory address
    bool allocated;         // in-use flag
    Block *prev;            // prev block if split from a larger allocation
    Block *next;            // next block if split from a larger allocation
    int event_count;        // number of outstanding CUDA events
    int gc_count;           // counter for prioritizing older / less useful blocks for
                            // garbage collection
    std::unique_ptr<History> history;
    History *history_last;

    Block(int device, cudaStream_t stream, size_t size, BlockPool *pool, void *ptr)
        : device(device)
        , stream(stream)
        , stream_uses()
        , size(size)
        , pool(pool)
        , ptr(ptr)
        , allocated(0)
        , prev(nullptr)
        , next(nullptr)
        , event_count(0)
        , gc_count(0)
    {
    }

    // constructor for search key
    Block(int device, cudaStream_t stream, size_t size)
        : device(device)
        , stream(stream)
        , stream_uses()
        , size(size)
        , pool(nullptr)
        , ptr(nullptr)
        , allocated(0)
        , prev(nullptr)
        , next(nullptr)
        , event_count(0)
        , gc_count(0)
    {
    }

    bool is_split() const
    {
        return (prev != nullptr) || (next != nullptr);
    }
};

static bool BlockComparator(const Block *a, const Block *b)
{
    if (a->stream != b->stream) {
        return (uintptr_t)a->stream < (uintptr_t)b->stream;
    }
    if (a->size != b->size) {
        return a->size < b->size;
    }
    return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

struct AllocParams {
    // for set range.
    Block search_key;
    BlockPool *pool;
    size_t alloc_size;
    Block *block;
    StatTypes stat_types = {false};
    cudaError_t err;

    AllocParams(int device, size_t size, cudaStream_t stream, BlockPool *pool, size_t alloc_size, DeviceStats &stats)
        : search_key(device, stream, size)
        , pool(pool)
        , alloc_size(alloc_size)
        , block(nullptr)
        , err(cudaSuccess)
    {
    }

    int device() const
    {
        return search_key.device;
    }
    cudaStream_t stream() const
    {
        return search_key.stream;
    }
    size_t size() const
    {
        return search_key.size;
    }
};

static std::string format_size(uint64_t size);

/* Add some tests */
void testDeviceCachingAllocator();
void testDeviceCachingAllocatorE2E();
void testDeviceCachingAllocatorSmallManagement();
void testDeviceCachingAllocatorFragment();

#endif
