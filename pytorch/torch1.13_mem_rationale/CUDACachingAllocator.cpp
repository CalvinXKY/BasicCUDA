/**
 *  The original code: https://github.com/pytorch/pytorch/blob/v1.13.1/c10/cuda/CUDACachingAllocator.cpp
 *  Modified for independent running.
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 * */
#include "CUDACachingAllocator.h"
#include "llvmMathExtras.h"

using namespace c10;

static std::string format_size(uint64_t size)
{
    std::ostringstream os;
    os.precision(2);
    os << std::fixed;
    if (size <= 1024) {
        os << size << " bytes";
    } else if (size <= 1048576) {
        os << (size / 1024.0);
        os << " KiB";
    } else if (size <= 1073741824ULL) {
        os << size / 1048576.0;
        os << " MiB";
    } else {
        os << size / 1073741824.0;
        os << " GiB";
    }
    return os.str();
}

//...device stat opts  begin...//
void update_stat(Stat &stat, int64_t amount)
{
    stat.current += amount;
    stat.peak = std::max(stat.current, stat.peak);
    if (amount > 0) {
        stat.allocated += amount;
    }
    if (amount < 0) {
        stat.freed += -amount;
    }
}

void reset_accumulated_stat(Stat &stat)
{
    stat.allocated = 0;
    stat.freed = 0;
}

void reset_peak_stat(Stat &stat)
{
    stat.peak = stat.current;
}

template <typename Func> void for_each_selected_stat_type(const StatTypes &stat_types, Func f)
{
    // original style: for (const auto stat_type : c10::irange(stat_types.size())) {
    for (int i = 0; i < stat_types.size(); ++i) {
        if (stat_types[i]) {
            f(i);
        }
    }
}

void update_stat_array(StatArray &stat_array, int64_t amount, const StatTypes &stat_types)
{
    for_each_selected_stat_type(
        stat_types, [&stat_array, amount](size_t stat_type) { update_stat(stat_array[stat_type], amount); });
}
//...device stat opts  end...//

class CachingAllocatorConfig {
    // private -> public for data observing.
public:
    static size_t max_split_size()
    {
        return instance().m_max_split_size;
    }
    static double garbage_collection_threshold()
    {
        return instance().m_garbage_collection_threshold;
    }

    // This is used to round-up allocation size to nearest power of 2 divisions.
    // More description below in function roundup_power2_next_division
    // As ane example, if we want 4 divisions between 2's power, this can be done
    // using env variable: PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
    static size_t roundup_power2_divisions()
    {
        return instance().m_roundup_power2_divisions;
    }
    static size_t roundup_bypass_threshold()
    {
        return instance().m_roundup_bypass_threshold;
    }

    static CachingAllocatorConfig &instance()
    {
        static CachingAllocatorConfig *s_instance = ([]() {
            auto inst = new CachingAllocatorConfig();
            const char *env = getenv("PYTORCH_CUDA_ALLOC_CONF");
            inst->parseArgs(env);
            return inst;
        })();
        return *s_instance;
    }

    void parseArgs(const char *env)
    {
        // If empty, set the default values
        m_max_split_size = std::numeric_limits<size_t>::max();
        m_roundup_power2_divisions = 0;
        m_roundup_bypass_threshold = std::numeric_limits<size_t>::max();
        m_garbage_collection_threshold = 0;

        if (env == nullptr) {
            return;
        }

        const std::string config(env);

        std::regex exp("[\\s,]+");
        std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
        std::sregex_token_iterator end;
        std::vector<std::string> options(it, end);

        for (auto option : options) {
            std::regex exp2("[:]+");
            std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
            std::sregex_token_iterator end2;
            std::vector<std::string> kv(it2, end2);
            if (kv.size() >= 2) {
                /* Maximum split size in MB.  Limited to large size blocks */
                if (kv[0].compare("max_split_size_mb") == 0) {
                    size_t val2 = stoi(kv[1]);
                    TORCH_CHECK(val2 > kLargeBuffer / (1024 * 1024),
                                "CachingAllocator option max_split_size_mb too small, must be > ",
                                kLargeBuffer / (1024 * 1024), "");
                    val2 = std::max(val2, kLargeBuffer / (1024 * 1024));
                    val2 = std::min(val2, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
                    m_max_split_size = val2 * 1024 * 1024;
                } else if (kv[0].compare("roundup_power2_divisions") == 0) {
                    size_t val2 = stoi(kv[1]);
                    TORCH_CHECK(llvm::isPowerOf2_64(val2), "For roundups, the divisons has to be power of 2 ", "");
                    m_roundup_power2_divisions = val2;
                } else if (kv[0].compare("roundup_bypass_threshold_mb") == 0) {
                    size_t val2 = stoi(kv[1]);
                    m_roundup_bypass_threshold = val2 * 1024 * 1024;
                } else if (kv[0].compare("garbage_collection_threshold") == 0) {
                    /*
                     * Perform garbage collection of GPU memory blocks to avoid
                     * triggering expensive sync-and-reclaim-all operation. Upon setting
                     * the threshold (e.g., 0.8), the allocator will start reclaiming
                     * blocks if GPU memory capacity usage exceeds the threshold (i.e.,
                     * 80% of total memory).
                     * Values 0.0 and 1.0 are not allowed as they are less meaningful.
                     */
                    double val2 = stod(kv[1]);
                    TORCH_CHECK(val2 > 0, "garbage_collect_threshold too small, set it 0.0~1.0", "");
                    TORCH_CHECK(val2 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0", "");
                    m_garbage_collection_threshold = val2;
                } else {
                    TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", kv[0]);
                }
            }
        }
    }

private:
    CachingAllocatorConfig()
        : m_max_split_size(std::numeric_limits<size_t>::max())
        , m_roundup_power2_divisions(0)
        , m_garbage_collection_threshold(0)
    {
    }
    std::atomic<size_t> m_max_split_size;
    std::atomic<size_t> m_roundup_power2_divisions;
    std::atomic<size_t> m_roundup_bypass_threshold;
    std::atomic<double> m_garbage_collection_threshold;
};

class DeviceCachingAllocator {
    // private -> public for data observing.
public:
    // lock around all operations
    mutable std::recursive_mutex mutex;

    // device statistics
    DeviceStats stats;

    // pool for unused block。
    // unallocated cached blocks larger than 1 MB
    BlockPool large_blocks;

    // unallocated cached blocks 1 MB or smaller
    BlockPool small_blocks;

    // allocated or in use by a stream. Holds all active allocations,
    // whether they came from graph_pools or one of the BlockPools above.
    set<Block *> active_blocks;

    // captures_underway tracks if a capture might be underway on any stream.
    // Most of the time it's zero, in which case malloc can avoid calling
    // cudaStreamGetCaptureInfo in the hot path.
    int captures_underway = 0;

    // record used memory.
    size_t total_allocated_memory = 0;

    size_t allowed_memory_maximum = 0;

    bool set_fraction = false;

public:
    DeviceCachingAllocator()
        : large_blocks(BlockComparator, /*is_small=*/false)
        , small_blocks(BlockComparator, /*is_small=*/true)
    {
        stats.max_split_size = CachingAllocatorConfig::max_split_size();
    }

    // This function takes the size and number of divisions argument and rounds
    // up the size argument for the nearest power-of-2 division.
    // For example, if we need to round-up 1200 and number of divisions is 4,
    // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
    // them, the values are 1024, 1280, 1536, and 1792. So the function will
    // return 1280 as the nearest ceiling of power-2 divison.

    static size_t roundup_power2_next_division(size_t size, size_t divisions)
    {
        // C10_UNLIKELY(size <= 4 || divisions <= 1)
        if (size <= 4 || divisions <= 1) {
            return size;
        }
        if (llvm::isPowerOf2_64(size)) {
            return size;
        }

        // divide the space between these 2's power into equal divisions
        // If division is zero, return the power-of-2 ceiling.
        size_t power2_floor = llvm::PowerOf2Floor(size);
        size_t power2_divison = power2_floor >> (63 - llvm::countLeadingZeros(divisions));
        if (power2_divison == 0) {
            return (power2_floor << 1);
        }
        size_t round_size_floor = size & (~(power2_divison - 1));
        return (round_size_floor == size) ? size : round_size_floor + power2_divison;
    }

    static size_t round_size(size_t size)
    {
        if (size < kMinBlockSize) {
            return kMinBlockSize;
        } else if (size > CachingAllocatorConfig::roundup_bypass_threshold()) {
            return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
        } else {
            auto divisions = CachingAllocatorConfig::roundup_power2_divisions();
            if (divisions > 0 && size > (kMinBlockSize * divisions)) {
                return roundup_power2_next_division(size, divisions);
            } else {
                return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
            }
        }
    }

    BlockPool &get_pool(size_t size, cudaStream_t stream)
    {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
        // captures_underway is a conservative guess that the current stream may be
        // capturing. It's only > 0 if some thread has begun and not yet ended a
        // capture, so it's usually 0, and we can short-circuit
        // cudaStreamCaptureStatus (which does a TLS lookup).
        if (C10_UNLIKELY(captures_underway)) {
            CaptureId_t id;
            cudaStreamCaptureStatus status;
            C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &id));
            if (status != cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
                TORCH_INTERNAL_ASSERT(status != cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated);
                // Retrieves the private pool assigned to this capture.
                auto it0 = capture_to_pool_map.find(id);
                TORCH_INTERNAL_ASSERT(it0 != capture_to_pool_map.end());
                auto it1 = graph_pools.find(it0->second);
                TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
                if (size <= kSmallSize) {
                    return it1->second->small_blocks;
                } else {
                    return it1->second->large_blocks;
                }
            }
        }
#endif
        if (size <= kSmallSize) {
            return small_blocks;
        } else {
            return large_blocks;
        }
    }

    static size_t get_allocation_size(size_t size)
    {
        if (size <= kSmallSize) {
            return kSmallBuffer;
        } else if (size < kMinLargeAlloc) {
            return kLargeBuffer;
        } else {
            return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
        }
    }

    StatType get_stat_type_for_pool(const BlockPool &pool)
    {
        return pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL;
    }

    // search block in pools -> found best block if it has  -> create a new block if it hasn't.
    Block *malloc(int device, size_t orig_size, cudaStream_t stream)
    {
        // mutex create:
        std::unique_lock<std::recursive_mutex> lock(mutex);
        // block info create：
        size_t size = round_size(orig_size);                 //  rounded to times of 512；
        auto &pool = get_pool(size, stream);                 //
        const size_t alloc_size = get_allocation_size(size); // alloc size suggestion。
        AllocParams params(device, size, stream, &pool, alloc_size, stats);
        // change stat_types
        params.stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
        params.stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;

        // First, try to get a block from the existing pool.
        bool block_found =
            // Search pool
            get_free_block(params)
            // Trigger callbacks and retry search
            || (trigger_free_memory_callbacks(params) && get_free_block(params));

        if (!block_found) {
            // Do garbage collection if the flag is set.
            if (C10_UNLIKELY(set_fraction && CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
                garbage_collect_cached_blocks();
            }
            // Attempt allocate
            block_found = alloc_block(params, false)
                          // Free enough available cached blocks to satisfy alloc and retry
                          // alloc.
                          || (release_available_cached_blocks(params) && alloc_block(params, false))
                          // Free all non-split cached blocks and retry alloc.
                          ||
                          (C10_LIKELY(captures_underway == 0) && release_cached_blocks() && alloc_block(params, true));

            if (!block_found) {
                // For any error code other than cudaErrorMemoryAllocation,
                // alloc_block should have thrown an exception already.
                TORCH_INTERNAL_ASSERT(params.err == cudaErrorMemoryAllocation);

                size_t device_free;
                size_t device_total;
                C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
                std::string allowed_info;

                if (set_fraction) {
                    allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
                }

                stats.num_ooms += 1;

                // "total capacity": total global memory on GPU
                // "allowed": memory is allowed to use, which set by fraction.
                // "already allocated": memory allocated by the program using the
                //                      caching allocator
                // "free": free memory as reported by the CUDA API
                // "cached": memory held by the allocator but not used by the program
                //
                // The "allocated" amount  does not include memory allocated outside
                // of the caching allocator, such as memory allocated by other programs
                // or memory held by the driver.
                //
                // The sum of "allocated" + "free" + "cached" may be less than the
                // total capacity due to memory held by the driver and usage by other
                // programs.
                //
                // Note that at this point free_cached_blocks has already returned all
                // possible "cached" memory to the driver. The only remaining "cached"
                // memory is split from a larger block that is partially in-use.
                TORCH_CHECK_WITH(false, "CUDA out of memory. Tried to allocate ", format_size(alloc_size), " (GPU ",
                                 device, "; ", format_size(device_total), " total capacity; ",
                                 format_size(stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
                                 " already allocated; ", format_size(device_free), " free; ", allowed_info,
                                 format_size(stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
                                 " reserved in total by PyTorch)",
                                 " If reserved memory is >> allocated memory try setting max_split_size_mb to avoid"
                                 " fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF",
                                 "");
            }
        }

        TORCH_INTERNAL_ASSERT(params.err == cudaSuccess && params.block != nullptr && params.block->ptr != nullptr);
        Block *block = params.block;
        Block *remaining = nullptr;

        const bool already_split = block->is_split();
        if (should_split(block, size)) {
            remaining = block;

            block = new Block(device, stream, size, &pool, block->ptr);
            block->prev = remaining->prev;
            if (block->prev) {
                block->prev->next = block;
            }
            block->next = remaining;

            remaining->prev = block;
            remaining->ptr = static_cast<char *>(remaining->ptr) + size;
            remaining->size -= size;
            bool inserted = pool.blocks.insert(remaining).second;
            // TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

            // if (context) {
            //   trimHistoryBefore(remaining, (char*)block->ptr + size);
            // }

            if (already_split) {
                // An already-split inactive block is being shrunk by size bytes.
                update_stat_array(stats.inactive_split_bytes, -block->size, params.stat_types);
            } else {
                // A new split inactive block is being created from a previously unsplit
                // block, size remaining->size bytes.
                for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
                    update_stat(stats.inactive_split_bytes[stat_type], remaining->size);
                    update_stat(stats.inactive_split[stat_type], 1);
                });
            }

        } else if (already_split) {
            // An already-split block is becoming active
            for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
                update_stat(stats.inactive_split_bytes[stat_type], -block->size);
                update_stat(stats.inactive_split[stat_type], -1);
            });
        }

        block->allocated = true;
        // if (context) {
        //   trimHistoryBefore(block, (char*)block->ptr + size);
        //   block->history = std::make_unique<History>(History{
        //       block->ptr,
        //       orig_size,
        //       std::move(context),
        //       std::move(block->history)});
        //   if (!block->history_last) {
        //     block->history_last = block->history.get();
        //   }
        // }

        bool inserted = active_blocks.insert(block).second;
        // TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
            update_stat(stats.allocation[stat_type], 1);
            update_stat(stats.allocated_bytes[stat_type], block->size);
            update_stat(stats.active[stat_type], 1);
            update_stat(stats.active_bytes[stat_type], block->size);
        });
        if (block->size >= CachingAllocatorConfig::max_split_size())
            update_stat(stats.oversize_allocations, 1);

        // c10::reportMemoryUsageToProfiler(
        //     block->ptr,
        //     block->size,
        //     stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        //     stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        //     c10::Device(c10::DeviceType::CUDA, device));

        return block;
    }

    // Dose not invoke cudaFree。
    void free(Block *block)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);

        block->allocated = false;

        // following logic might modifying underlaying Block, causing the size
        // changed. We store ahead for reporting
        auto orig_block_ptr = block->ptr;
        auto orig_block_size = block->size;

        StatTypes stat_types = {false};
        stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
        stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
        for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
            update_stat(stats.allocation[stat_type], -1);
            update_stat(stats.allocated_bytes[stat_type], -block->size);
        });
        if (block->size >= CachingAllocatorConfig::max_split_size())
            update_stat(stats.oversize_allocations, -1);

        if (!block->stream_uses.empty()) {
            if (C10_UNLIKELY(captures_underway)) {
                // It's forbidden to cudaEventQuery an event recorded during CUDA graph
                // capture. We conservatively defer recording end-of-life events until
                // the next call to process_events() (which won't happen until no
                // captures are underway)

                // needs_events_deferred_until_no_capture.push_back(block);
            } else {
                // insert_events(block);
            }
        } else {
            free_block(block);
        }

        // c10::reportMemoryUsageToProfiler(
        //     orig_block_ptr,
        //     -orig_block_size,
        //     stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        //     stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        //     c10::Device(c10::DeviceType::CUDA, block->device));
    }

    void free_block(Block *block)
    {
        TORCH_INTERNAL_ASSERT(!block->allocated && block->event_count == 0 && block->stream_uses.empty());

        size_t original_block_size = block->size;

        auto &pool = *block->pool;
        int64_t net_change_inactive_split_blocks = 0;
        int64_t net_change_inactive_split_size = 0;

        const std::array<Block *, 2> merge_candidates = {block->prev, block->next};
        for (Block *merge_candidate : merge_candidates) {
            const int64_t subsumed_size = try_merge_blocks(block, merge_candidate, pool);
            if (subsumed_size > 0) {
                net_change_inactive_split_blocks -= 1;
                net_change_inactive_split_size -= subsumed_size;
            }
        }

        active_blocks.erase(block);
        // Makes sure the Block* isn't already present in the pool we're freeing it
        // back into.
        bool inserted = pool.blocks.insert(block).second;
        TORCH_INTERNAL_ASSERT(inserted);

        if (block->is_split()) {
            net_change_inactive_split_blocks += 1;
            net_change_inactive_split_size += block->size;
        }

        StatTypes stat_types = {false};
        stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
        stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;
        for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
            update_stat(stats.inactive_split[stat_type], net_change_inactive_split_blocks);
            update_stat(stats.inactive_split_bytes[stat_type], net_change_inactive_split_size);
            update_stat(stats.active[stat_type], -1);
            update_stat(stats.active_bytes[stat_type], -original_block_size);
        });
    }

    /** combine previously split blocks. returns the size of the subsumed block,
     * or 0 on failure. */
    size_t try_merge_blocks(Block *dst, Block *src, BlockPool &pool)
    {
        if (!src || src->allocated || src->event_count > 0 || !src->stream_uses.empty()) {
            return 0;
        }

        TORCH_CHECK(dst->is_split() && src->is_split());

        if (dst->prev == src) { // [src dst]
            dst->ptr = src->ptr;
            dst->prev = src->prev;
            if (dst->prev) {
                dst->prev->next = dst;
            }
            if (!dst->history) {
                dst->history = std::move(src->history);
                dst->history_last = src->history_last;
            } else if (src->history) {
                src->history_last->next = std::move(dst->history);
                dst->history = std::move(src->history);
            }
            src->history_last = nullptr;
        } else { // [dest src]
            dst->next = src->next;
            if (dst->next) {
                dst->next->prev = dst;
            }

            if (!dst->history) {
                dst->history = std::move(src->history);
                dst->history_last = src->history_last;
            } else if (src->history) {
                dst->history_last->next = std::move(src->history);
                dst->history_last = src->history_last;
            }
            src->history_last = nullptr;
        }
        const size_t subsumed_size = src->size;
        dst->size += subsumed_size;
        auto erased = pool.blocks.erase(src);
        // TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
        delete src;

        return subsumed_size;
    }

    bool get_free_block(AllocParams &p)
    {
        BlockPool &pool = *p.pool;

        auto it = pool.blocks.lower_bound(&p.search_key); // set-container search, return minium satisfied value.
        if (it == pool.blocks.end() || (*it)->stream != p.stream())
            return false;

        // Do not return an oversized block for a large request
        if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
            ((*it)->size >= CachingAllocatorConfig::max_split_size()))
            return false;
        // Allow oversized block size to be rounded up but within a limit
        if ((p.size() >= CachingAllocatorConfig::max_split_size()) && ((*it)->size >= p.size() + kLargeBuffer))
            return false;
        p.block = *it;
        (*it)->gc_count = 0; // Denote this block has been used
        pool.blocks.erase(it);
        return true;
    }

    // only one invoking
    bool trigger_free_memory_callbacks(AllocParams &p)
    {
        bool freed_memory = false;
        // code commented
        // for (const auto& name : FreeCudaMemoryCallbacksRegistry()->Keys()) {
        //   freed_memory |=
        //       FreeCudaMemoryCallbacksRegistry()->Create(name)->Execute();
        // }
        return freed_memory;
    }

    void garbage_collect_cached_blocks()
    {
        // Free unused cached blocks to reclaim GPU memory.
        // Unlike release_cached_blocks(), this does not enforce synchronization and
        // therefore should be of less overheads.

        size_t gc_threshold =
            static_cast<size_t>(CachingAllocatorConfig::garbage_collection_threshold() * allowed_memory_maximum);
        // No need to trigger GC yet
        if (total_allocated_memory <= gc_threshold) {
            return;
        }
        const auto target_size = total_allocated_memory - gc_threshold;
        size_t gc_reclaimed = 0;

        // Calculate the total age of the free-able blocks. We'll use it later to
        // get "avg age" threshold.
        double total_age = 0.0;
        int freeable_block_count = 0;
        for (auto &b : large_blocks.blocks) {
            if (!b->is_split()) {
                total_age += b->gc_count;
                ++freeable_block_count;
            }
        }
        // No free-able blocks?
        if (freeable_block_count == 0) {
            return;
        }

        // Repeat GC until we reach reclaim > target size.
        bool block_freed = true;
        while (gc_reclaimed < target_size && block_freed == true && freeable_block_count > 0) {
            // Free blocks exceeding this age threshold first.
            double age_threshold = total_age / freeable_block_count;
            // Stop iteration if we can no longer free a block.
            block_freed = false;

            // Free blocks of > avg age. Don't stop upon reaching the target_size,
            // we don't want this GC to be triggered frequently.
            auto it = large_blocks.blocks.begin();
            while (it != large_blocks.blocks.end()) {
                Block *block = *it;
                ++it;
                if (!block->is_split() && block->gc_count >= age_threshold) {
                    block_freed = true;
                    gc_reclaimed += block->size;
                    total_age -= block->gc_count; // Decrement the age
                    freeable_block_count--;       // One less block that can be freed
                    release_block(block);
                }
            }
        }
    }

    bool alloc_block(AllocParams &p, bool isRetry)
    {
        // Defensively checks for preexisting CUDA error state.
        C10_CUDA_CHECK(cudaGetLastError());

        size_t size = p.alloc_size;
        void *ptr;

        if (isRetry) {
            stats.num_alloc_retries += 1;
        }

        if (set_fraction && total_allocated_memory + size > allowed_memory_maximum) {
            p.err = cudaErrorMemoryAllocation;
            return false;
        } else {
            // origin： p.err = cudaMallocMaybeCapturing(&ptr, size);
            p.err = cudaMalloc(&ptr, size);
            if (p.err != cudaSuccess) {
                if (p.err == cudaErrorMemoryAllocation) {
                    // If this is the first attempt (!isRetry), we can forgive and clear
                    // CUDA's
                    //   internal error state.
                    // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH
                    // will take
                    //   over to throw a helpful exception. The user can choose to catch
                    //   the exception, free some stuff in their script, and attempt their
                    //   allocation again. In this case, we can also forgive and clear
                    //   CUDA's internal error state.
                    cudaGetLastError();
                } else {
                    // If the error's unrelated to memory allocation, we should throw
                    // immediately.
                    C10_CUDA_CHECK(p.err);
                }
                return false;
            }
        }

        // if (p.pool->owner_PrivatePool) {
        //   // The block is for a CUDA graph's PrivatePool.
        //   p.pool->owner_PrivatePool->cudaMalloc_count++;
        // }

        total_allocated_memory += size;
        p.block = new Block(p.device(), p.stream(), size, p.pool, (char *)ptr);
        for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
            update_stat(stats.segment[stat_type], 1);
            update_stat(stats.reserved_bytes[stat_type], size);
        });
        if (size >= CachingAllocatorConfig::max_split_size())
            update_stat(stats.oversize_segments, 1);

        // p.block came from new, not cudaMalloc. It should not be nullptr here.
        TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
        return true;
    }

    bool should_split(const Block *block, size_t size)
    {
        size_t remaining = block->size - size;
        if (block->pool->is_small) {
            return remaining >= kMinBlockSize;
        } else {
            return (size < CachingAllocatorConfig::max_split_size()) && (remaining > kSmallSize);
        }
    }

    /** Free one or more oversize blocks to the system allocator.  But only enough
     * **/
    /** to satisfy the target size **/
    // for  alloc()  emptyCache();
    bool release_available_cached_blocks(const AllocParams &p)
    {
        if (CachingAllocatorConfig::max_split_size() == std::numeric_limits<size_t>::max())
            return false;
        BlockPool &pool = *p.pool;

        // because of std::unique_ptr, block cannot be trivially copied
        Block key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool, p.search_key.ptr);
        key.size =
            (key.size < CachingAllocatorConfig::max_split_size()) ? CachingAllocatorConfig::max_split_size() : key.size;
        auto it = pool.blocks.lower_bound(&key);
        if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
            // No single block is large enough; free multiple oversize blocks,
            // starting with the largest
            if (it == pool.blocks.begin())
                return false;
            size_t totalReleased = 0;
            --it; // Back up one item.  Now on the largest block for the correct
                  // stream
            while ((totalReleased < key.size) && ((*it)->size >= CachingAllocatorConfig::max_split_size()) &&
                   ((*it)->stream == p.stream())) {
                auto cur = it;
                totalReleased += (*it)->size;
                if (it != pool.blocks.begin()) {
                    --it;
                    release_block(*cur);
                } else {
                    release_block(*cur);
                    break;
                }
            }
            if (totalReleased < key.size)
                return false;
        } else {
            release_block(*it);
        }
        return true;
    }

    bool release_cached_blocks()
    {
        // First ensure that all blocks that can't currently be allocated due to
        // outstanding events are returned to the pool.
        // synchronize_and_free_events();

        // Free all non-split cached blocks to system allocator
        release_blocks(large_blocks);
        release_blocks(small_blocks);

        // for (auto it = graph_pools_freeable.begin();
        //      it != graph_pools_freeable.end();) {
        //   // See notifyCaptureDestroy for the strategy here.
        //   TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
        //   release_blocks(it->second->small_blocks);
        //   release_blocks(it->second->large_blocks);
        //   if (it->second->cudaMalloc_count == 0) {
        //     auto erase_count = graph_pools.erase(it->first);
        //     TORCH_INTERNAL_ASSERT(erase_count == 1);
        //     it = graph_pools_freeable.erase(it);
        //   } else {
        //     ++it;
        //   }
        // }

        return true;
    }

    /*
     * Do not invoke release_block() without if(!block->prev && !block->next).
     * It will raise a segment error, if release parts of segment.
     */
    void release_block(Block *block)
    {
        C10_CUDA_CHECK(cudaFree((void *)block->ptr));
        total_allocated_memory -= block->size;
        auto *pool = block->pool;

        // if (pool->owner_PrivatePool) {
        //   // The cudaFreed block belonged to a CUDA graph's PrivatePool.
        //   TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->cudaMalloc_count > 0);
        //   pool->owner_PrivatePool->cudaMalloc_count--;
        // }

        StatTypes stat_types = {false};
        stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
        stat_types[static_cast<size_t>(get_stat_type_for_pool(*pool))] = true;
        for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
            update_stat(stats.segment[stat_type], -1);
            update_stat(stats.reserved_bytes[stat_type], -block->size);
        });
        if (block->size >= CachingAllocatorConfig::max_split_size())
            update_stat(stats.oversize_segments, -1);

        pool->blocks.erase(block);
        delete block;
    }

    void release_blocks(BlockPool &pool)
    {
        // Frees all non-split blocks
        auto it = pool.blocks.begin();
        while (it != pool.blocks.end()) {
            Block *block = *it;
            ++it;
            if (!block->prev && !block->next) {
                release_block(block);
            }
        }
    }

    /** returns cached blocks to the system allocator **/
    void emptyCache()
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        release_cached_blocks();
    }
};

void auxPrintPtrInfo(void *ptr)
{
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
    if (err == cudaSuccess) {
        if (attributes.type == cudaMemoryTypeDevice) {
            printf("Pointer 0x%x is a device pointer.\n", ptr);
        } else if (attributes.type == cudaMemoryTypeHost) {
            printf("Pointer 0x%x is a host pointer.\n", ptr);
        } else if (attributes.type == cudaMemoryTypeManaged) {
            printf("Pointer 0x%x is a managed pointer.\n", ptr);
        } else {
            printf("Pointer 0x%x is an unregistered cuda pointer.\n", ptr);
        }
    } else {
        printf("cudaPointerGetAttributes() failed: %s\n", cudaGetErrorString(err));
    }
}

void auxPrintPoolBlocksInfo(DeviceCachingAllocator &allocator, string str)
{
    cout << str << " Print allocator pools info:" << endl
         << "  1> The block in large_blocks, number: " << allocator.large_blocks.blocks.size() << endl;
    int idx = 0;
    for (const auto &block : allocator.large_blocks.blocks) {
        printf("   Ptr%d: 0x%x, data ptr: 0x%x, data size: %s, is_split: %d\n", idx++, block, block->ptr,
               format_size(block->size).c_str(), block->is_split());
    }
    idx = 0;
    cout << "  2> The block in small_blocks, number: " << allocator.small_blocks.blocks.size() << endl;
    for (const auto &block : allocator.small_blocks.blocks) {
        printf("   Ptr%d: 0x%x, data ptr: 0x%x, data size: %s, is_split: %d\n", idx++, block, block->ptr,
               format_size(block->size).c_str(), block->is_split());
    }
    cout << "  3> The block in active_blocks, number: " << allocator.active_blocks.size() << endl;
    idx = 0;
    for (const auto &block : allocator.active_blocks) {
        printf("   Ptr%d: 0x%x, data ptr: 0x%x, data size: %s, is_split: %d\n", idx++, block, block->ptr,
               format_size(block->size).c_str(), block->is_split());
    }
}

void testDeviceCachingAllocator()
{
    DeviceCachingAllocator allocator;
    cout << "=====================round_size fucntion test:==========================" << endl;
    for (int i = 0; i < 500000; i += 50000) {
        size_t round_size = allocator.round_size(i);
        cout << "input size:" << i << " format:" << format_size(i);
        cout << ";round size:" << round_size << "; format:" << format_size(round_size) << endl;
    }
    cout << endl;

    cout << "====================get_allocation_size fucntion test:==================" << endl;
    cout << "========Mini gap======" << endl;
    for (int i = 0; i < 1000000; i += 50000) {
        size_t allocation_size = allocator.get_allocation_size(allocator.round_size(i));
        cout << "input size:" << i << " format:" << format_size(i);
        cout << "; round size:" << allocation_size << "; format:" << format_size(allocation_size) << endl;
    }
    cout << "========Small gap======" << endl;
    for (int i = 0; i < 10000000; i += 500000) {
        size_t allocation_size = allocator.get_allocation_size(allocator.round_size(i));
        cout << "input size:" << i << " format:" << format_size(i);
        cout << "; round size:" << allocation_size << "; format:" << format_size(allocation_size) << endl;
    }
    cout << "=========Middle gap=====" << endl;
    for (int i = 0; i < 100000000; i += 5000000) {
        size_t allocation_size = allocator.get_allocation_size(allocator.round_size(i));
        cout << "input size:" << i << " format:" << format_size(i);
        cout << "; round size:" << allocation_size << "; format:" << format_size(allocation_size) << endl;
    }
    cout << endl;

    cout << "====================get_free_block fucntion test:==================" << endl;
    BlockPool block_pool(BlockComparator, true);
    vector<Block *> block_vec;
    cudaStream_t cuda_stream = cudaStreamDefault;
    DeviceStats stats;
    int cur_device = 0;
    for (int i = 1000; i < 1000000; i += 10000) {
        Block *block_temp = new Block(cur_device, cuda_stream, allocator.round_size(i), &block_pool, nullptr);
        block_vec.push_back(block_temp);
        block_pool.blocks.insert(block_temp);
    }
    for (int i = 24340; i < 100000; i += 10000) {
        size_t orig_size = i;
        cout << "The original request size: " << orig_size;
        size_t size = allocator.round_size(orig_size);
        const size_t alloc_size = allocator.get_allocation_size(size);
        AllocParams params(cur_device, size, cuda_stream, &block_pool, alloc_size, stats);
        cout << " block_found=" << allocator.get_free_block(params);
        cout << " block size=" << params.block->size << endl;
    }

    cout << "====================malloc test:==================" << endl;
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    Block *block_tmp = allocator.malloc(device, 100002, cuda_stream);
    cout << "Block info: device:" << block_tmp->device << " size:" << block_tmp->size
         << " block pool: " << block_tmp->pool << endl;
    auxPrintPtrInfo(block_tmp->ptr);
    auxPrintPoolBlocksInfo(allocator, "Created a block.");

    cout << "====================free test:==================" << endl;
    allocator.free(block_tmp);
    cout << "Block: " << block_tmp << " was freed. But it is still in device." << endl;
    auxPrintPtrInfo(block_tmp->ptr);
    allocator.emptyCache();
    auxPrintPoolBlocksInfo(allocator, "Relased the block.");
}

void testDeviceCachingAllocatorE2E()
{
    // end2end test:
    // create blocks
    cout << "====================testDeviceCachingAllocatorE2E test:==================" << endl;
    DeviceCachingAllocator allocator;
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    /* Apply for <kSmallSize(1048576) memory twice, It would create a 2M block to small blocks first，
      then which would be splited.
     The first activeblock is tmp1, its pre ptr is nullptr, next ptr is tmp2.
     The second activeblock is tmp2, its pre ptr is tmp1, next ptr is the remaining block in small_blocks.
    */
    Block *block_tmp1 = allocator.malloc(device, 100002, cudaStreamDefault);
    auxPrintPoolBlocksInfo(allocator, "Apply for 100002B memory block.");
    Block *block_tmp2 = allocator.malloc(device, 1048575, cudaStreamDefault);
    auxPrintPoolBlocksInfo(allocator, "Apply for 1048575B memory block.");

    /* The small_blocks pool does not satisfy request, so it would create new 2Mblock, then split it*/
    Block *block_tmp3 = allocator.malloc(device, 1000000, cudaStreamDefault);
    auxPrintPoolBlocksInfo(allocator, "Apply for other 1097152B memory block.");

    /* Apply for 4M block(>1M), allcator would create 20M block in the large blocks, then split it */

    Block *block_tmp4 = allocator.malloc(device, 2097152 * 2, cudaStreamDefault);
    auxPrintPoolBlocksInfo(allocator, "Apply for 4M block.");

    /* Apply for a block 12M (>10M), they would split the block in the large pool. */
    Block *block_tmp5 = allocator.malloc(device, 1048576 * 11, cudaStreamDefault);
    auxPrintPoolBlocksInfo(allocator, "Apply for 11M block.");
    allocator.free(block_tmp5);
    auxPrintPoolBlocksInfo(allocator, "After free 11M block.");

    /* Apply for 21M block(>20M), allcator would create 21M block in the active block.
    The block's is_split=0
    After the block(tmp) released, it will be marked to large blocks pool
    */
    Block *block_tmp6 = allocator.malloc(device, 1048576 * 21, cudaStreamDefault);
    auxPrintPoolBlocksInfo(allocator, "Apply for 21M block.");
    allocator.free(block_tmp6);
    auxPrintPoolBlocksInfo(allocator, "After free 21M block.");
}

void testDeviceCachingAllocatorSmallManagement()
{
    cout << "====================testDeviceCachingAllocatorSmallManagement test:==================" << endl;
    DeviceCachingAllocator allocator;
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    /* Apply for <kSmallSize(1048575) memory twice, It would create a 2M block to small blocks first，
      then which would be splited.
     The first activeblock is tmp1, its pre ptr is nullptr, next ptr is tmp2.
     The second activeblock is tmp2, its pre ptr is tmp1, next ptr is the remaining block in small_blocks.
    */
    Block *block_tmp1 = allocator.malloc(device, 1048575, cudaStreamDefault);
    auxPrintPoolBlocksInfo(allocator, "Apply for 1M memory block.");
    Block *block_tmp2 = allocator.malloc(device, 1048575, cudaStreamDefault);
    /* After these two applications, the 2M block was used, them it would remove from small blocks pool*/
    auxPrintPoolBlocksInfo(allocator, "Apply for 1M memory block.");

    /* After released tmp2, the unused block would be mark to small blocks pool*/
    allocator.free(block_tmp2);
    auxPrintPoolBlocksInfo(allocator, "After free 1M block.");

    /* After released tmp1, would trigger blocks merge operation. Then, It could apply for >1M block */
    allocator.free(block_tmp1);
    Block *block_tmp3 = allocator.malloc(device, 1048575 * 0.8, cudaStreamDefault);
    Block *block_tmp4 = allocator.malloc(device, 1048575 * 0.8, cudaStreamDefault);
    auxPrintPoolBlocksInfo(allocator, "Apply for 2 * 0.8M blocks.");
}

void testDeviceCachingAllocatorFragment()
{
    cout << "====================testDeviceCachingAllocatorFragment test:==================" << endl;
    DeviceCachingAllocator allocator;
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cout << "==================== Small fragments test:====================" << endl;
    vector<Block *> block_vec;
    /* Apply for 9 blocks, they would split from the 2M block. */
    for (int i = 0; i < 9; ++i) {
        Block *block_tmp = allocator.malloc(device, 104857 * 2, cudaStreamDefault);
        block_vec.push_back(block_tmp);
    }
    auxPrintPoolBlocksInfo(allocator, "Apply for 10 blocks for using.");
    /* Release some parts of blocks */
    // interval = 2, free some blocks
    for (int i = 0; i < 9; i += 2) {
        allocator.free(block_vec[i]);
    }
    auxPrintPoolBlocksInfo(allocator, "After free some blocks.");
    /* Then, create 500KB block.
    It will trigger a new 2M block segment, the remaining blocks can not be used.
    */
    Block *block_tmp_new = allocator.malloc(device, 104857 * 5, cudaStreamDefault);
    auxPrintPoolBlocksInfo(allocator, "After free some blocks.");

    /* Also, the emptyCache() would not clear the blocks in pool */
    allocator.emptyCache();
    auxPrintPoolBlocksInfo(allocator, "After empty cache option.");

    cout << "====================Big fragments test:====================" << endl;
    DeviceCachingAllocator allocator2;
    /* Apply for 9 blocks, they would split from the 20M block. */
    vector<Block *> block_vec_large;
    for (int i = 0; i < 9; ++i) {
        Block *block_tmp = allocator2.malloc(device, 1048576 * 2, cudaStreamDefault);
        block_vec_large.push_back(block_tmp);
    }
    auxPrintPoolBlocksInfo(allocator2, "Apply for 10 * 2M blocks.");
    /* Release some parts of blocks */
    // interval = 2, free some blocks
    for (int i = 0; i < 9; i += 2) {
        allocator2.free(block_vec_large[i]);
    }
    allocator.emptyCache();
    auxPrintPoolBlocksInfo(allocator2, "After free some blocks.");
}
