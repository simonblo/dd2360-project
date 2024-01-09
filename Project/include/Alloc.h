#ifndef Alloc_H
#define Alloc_H

#include <cstdio>
#include <cuda_runtime.h>

__host__ __device__
inline long get_idx(long x, long y, long stride_x)
{
    return x + (y * stride_x);
}

__host__ __device__
inline long get_idx(long x, long y, long z, long stride_y, long stride_z)
{
    return z + (y * stride_z) + (x * stride_y * stride_z);
}

template <class type>
inline type* newArr1(size_t sz1)
{
    type* arr = new type[sz1];
    return arr;
}

template <class type>
inline type** newArr2(size_t sz1, size_t sz2)
{
    type** arr = new type*[sz1];
    type* ptr = newArr1<type>(sz1 * sz2);
    for (size_t i = 0; i < sz1; i++)
    {
        arr[i] = ptr;
        ptr += sz2;
    }
    return arr;
}

template <class type>
inline type*** newArr3(size_t sz1, size_t sz2, size_t sz3)
{
    type*** arr = new type**[sz1];
    type** ptr = newArr2<type>(sz1 * sz2, sz3);
    for (size_t i = 0; i < sz1; i++)
    {
        arr[i] = ptr;
        ptr += sz2;
    }
    return arr;
}

/* Build chained pointer hierachy for pre-existing bottom level                        *
 * Provide a pointer to a contig. 1D memory region which was already allocated in "in" *
 * The function returns a pointer chain to which allows subscript access (x[i][j])     */

template <class type>
inline type** newArr2(type** in, size_t sz1, size_t sz2)
{
    *in = newArr1<type>(sz1 * sz2);
    type** arr = newArr1<type*>(sz1);
    type* ptr = *in;
    for (size_t i = 0; i < sz1; i++)
    {
        arr[i] = ptr;
        ptr += sz2;
    }
    return arr;
}

template <class type>
inline type*** newArr3(type** in, size_t sz1, size_t sz2, size_t sz3)
{
    *in = newArr1<type>(sz1 * sz2 * sz3);
    type*** arr = newArr2<type*>(sz1, sz2);
    type** arr2 = *arr;
    type* ptr = *in;
    size_t szarr2 = sz1 * sz2;
    for (size_t i = 0; i < szarr2; i++)
    {
        arr2[i] = ptr;
        ptr += sz3;
    }
    return arr;
}

template <class type>
inline type** newPinnedArr2(type** in, size_t sz1, size_t sz2)
{
    cudaHostAlloc(in, sizeof(type) * sz1 * sz2, cudaHostAllocDefault);
    type** arr = newArr1<type*>(sz1);
    type* ptr = *in;
    for (size_t i = 0; i < sz1; i++)
    {
        arr[i] = ptr;
        ptr += sz2;
    }
    return arr;
}

template <class type>
inline type*** newPinnedArr3(type** in, size_t sz1, size_t sz2, size_t sz3)
{
    cudaHostAlloc(in, sizeof(type) * sz1 * sz2 * sz3, cudaHostAllocDefault);
    type*** arr = newArr2<type*>(sz1, sz2);
    type** arr2 = *arr;
    type* ptr = *in;
    size_t szarr2 = sz1 * sz2;
    for (size_t i = 0; i < szarr2; i++)
    {
        arr2[i] = ptr;
        ptr += sz3;
    }
    return arr;
}

// methods to deallocate arrays
template <class type> inline void delArray1(type* arr)
{ delete[](arr); }
template <class type> inline void delArray2(type** arr)
{ delArray1(arr[0]); delete[](arr); }
template <class type> inline void delArray3(type*** arr)
{ delArray2(arr[0]); delete[](arr); }

// versions with dummy dimensions (for backwards compatibility)
template <class type> inline void delArr1(type* arr)
{ delArray1<type>(arr); }
template <class type> inline void delArr2(type** arr, size_t sz1)
{ delArray2<type>(arr); }
template <class type> inline void delArr3(type*** arr, size_t sz1, size_t sz2)
{ delArray3<type>(arr); }

// methods to deallocate pinned arrays
template <class type> inline void delPinnedArr1(type* arr)
{ cudaFree(arr); }
template <class type> inline void delPinnedArr2(type** arr)
{ delPinnedArr1(arr[0]); delete[](arr); }
template <class type> inline void delPinnedArr3(type*** arr)
{ delPinnedArr2(arr[0]); delete[](arr); }

#endif
