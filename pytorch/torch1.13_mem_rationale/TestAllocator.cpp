/**
 *  The cuda caching allocator tests
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 * */

#include "CUDACachingAllocator.h"

int main()
{
    testDeviceCachingAllocator();
    testDeviceCachingAllocatorE2E();
    testDeviceCachingAllocatorSmallManagement();
    testDeviceCachingAllocatorFragment();
    return 0;
}
