// ============================================================
// ripple_sort.h - Adaptive Radix + Introsort Hybrid
// // Version: v4.3.8 (Correctness & Cleanup)
// ============================================================
// A single-header drop-in sorting library for integer types.
// Features:
//   - Small-data fast path (pdqsort-style, n < 1024)
//   - Adaptive pre-scan (all-equal / sorted / reversed / nearly-sorted)
//   - Entropy-adaptive dispatch: counting / MSD+LSD / 8-bit / 11-bit radix
//   - Bitmap-skip counting sort for sparse distributions
//   - Stack-allocated small counting buckets (partial zeroing)
//   - MSD+LSD hybrid radix for cache-friendly medium-to-large inputs
//   - 11-bit x 3-pass single-threaded radix sort
//   - 8-bit x 4-pass multi-threaded radix sort for large inputs
//   - 8-pass 64-bit radix sort for long long
//   - Software prefetch on radix read loops (threshold-gated)
//   - Introsort fallback (quicksort + heapsort + insertion sort)
//   - Top-K (lazy sort)
//   - LazySortedView (zero-cost top-K view)
//   - Full ascending / descending support for all paths
// Works on C++17 and C++20.
// ============================================================

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER)
#   include <intrin.h>
#   include <xmmintrin.h>
#   define RIPPLE_PREFETCH(p) _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0)
#else
#   define RIPPLE_PREFETCH(p) __builtin_prefetch(p)
#endif

namespace ripple_sort {
    namespace detail {

        // ============================================================
        // Cross-platform bit utilities
        // ============================================================
        inline int ctz64(unsigned long long x)
        {
            if (x == 0ull) return 64;

#if defined(_MSC_VER)
            unsigned long idx;
#if defined(_M_X64) || defined(_M_ARM64)
            _BitScanForward64(&idx, x);
            return static_cast<int>(idx);
#else
            if (_BitScanForward(&idx, static_cast<unsigned long>(x & 0xFFFFFFFFull)))
                return static_cast<int>(idx);
            _BitScanForward(&idx, static_cast<unsigned long>(x >> 32));
            return static_cast<int>(idx) + 32;
#endif
#else
            return __builtin_ctzll(x);
#endif
        }

        inline int clz64(unsigned long long x)
        {
            if (x == 0ull) return 64;

#if defined(_MSC_VER)
            unsigned long idx;
#if defined(_M_X64) || defined(_M_ARM64)
            _BitScanReverse64(&idx, x);
            return 63 - static_cast<int>(idx);
#else
            if (x >> 32)
            {
                _BitScanReverse(&idx, static_cast<unsigned long>(x >> 32));
                return 31 - static_cast<int>(idx);
            }
            _BitScanReverse(&idx, static_cast<unsigned long>(x));
            return 63 - static_cast<int>(idx);
#endif
#else
            return __builtin_clzll(x);
#endif
        }

        // ============================================================
        // Compile-time constants (all tunable via -D)
        // ============================================================
#ifndef RIPPLE_INSERTION_SORT_THRESHOLD
#define RIPPLE_INSERTION_SORT_THRESHOLD 64
#endif
        constexpr int INSERTION_SORT_THRESHOLD = RIPPLE_INSERTION_SORT_THRESHOLD;

        constexpr int NINTHER_THRESHOLD = 128;

#ifndef RIPPLE_MAX_NEARLY_SORTED_INVERSION
#define RIPPLE_MAX_NEARLY_SORTED_INVERSION 16
#endif
        constexpr int MAX_NEARLY_SORTED_INVERSION = RIPPLE_MAX_NEARLY_SORTED_INVERSION;

#ifndef RIPPLE_PRESCAN_THRESHOLD
#define RIPPLE_PRESCAN_THRESHOLD 2048
#endif
        constexpr int PRESCAN_THRESHOLD = RIPPLE_PRESCAN_THRESHOLD;

        constexpr int PARTIAL_INSERTION_LIMIT = 8;
        constexpr int RADIX_SORT_THRESHOLD = 256;
        constexpr int SORT_NETWORK_THRESHOLD = 5;

#ifndef RIPPLE_SMALL_DATA_THRESHOLD
#define RIPPLE_SMALL_DATA_THRESHOLD 256
#endif
        constexpr int SMALL_DATA_THRESHOLD = RIPPLE_SMALL_DATA_THRESHOLD;

#ifndef RIPPLE_SMALL_INSERTION_THRESHOLD
#define RIPPLE_SMALL_INSERTION_THRESHOLD 32
#endif
        constexpr int SMALL_INSERTION_THRESHOLD = RIPPLE_SMALL_INSERTION_THRESHOLD;

        constexpr int MAX_SMALL_DEPTH = 32;

#ifndef RIPPLE_PARALLEL_RADIX_MIN
#define RIPPLE_PARALLEL_RADIX_MIN 2000000
#endif
        constexpr int PARALLEL_RADIX_MIN = RIPPLE_PARALLEL_RADIX_MIN;

#ifndef RIPPLE_MSD_LSD_THRESHOLD
#define RIPPLE_MSD_LSD_THRESHOLD 50000
#endif
        constexpr int MSD_LSD_THRESHOLD = RIPPLE_MSD_LSD_THRESHOLD;

#ifndef RIPPLE_QUICK_REJECT_SAMPLE
#define RIPPLE_QUICK_REJECT_SAMPLE 64
#endif
        constexpr int QUICK_REJECT_SAMPLE = RIPPLE_QUICK_REJECT_SAMPLE;

        constexpr unsigned long long COUNTING_SORT_RANGE_LIMIT = 1ull << 20;
        constexpr int COUNTING_STACK_BUCKETS = 4096;

        static_assert(COUNTING_STACK_BUCKETS <= 16384,
            "COUNTING_STACK_BUCKETS too large, may cause stack overflow");

        constexpr int MSD_BUCKET_BITS = 8;
        constexpr int MSD_BUCKET_SIZE = 1 << MSD_BUCKET_BITS;
        constexpr int MSD_BUCKET_MASK = MSD_BUCKET_SIZE - 1;
        constexpr int MSD_SHIFT = 24;
        constexpr int LSD12_SIZE = 1 << 12;
        constexpr int LSD12_MASK = LSD12_SIZE - 1;
        constexpr int PREFETCH_DISTANCE = 8;
        constexpr int PREFETCH_THRESHOLD = 16384;

        // ============================================================
        // Sorting networks
        // ============================================================
        template <typename T, typename Compare>
        inline void sort_network_3(T* data, Compare comp)
        {
            auto cswap = [&](int i, int j) { if (comp(data[j], data[i])) std::swap(data[i], data[j]); };
            cswap(0, 2); cswap(0, 1); cswap(1, 2);
        }

        template <typename T, typename Compare>
        inline void sort_network_4(T* data, Compare comp)
        {
            auto cswap = [&](int i, int j) { if (comp(data[j], data[i])) std::swap(data[i], data[j]); };
            cswap(0, 2); cswap(1, 3); cswap(0, 1); cswap(2, 3); cswap(1, 2);
        }

        template <typename T, typename Compare>
        inline void sort_network_5(T* data, Compare comp)
        {
            auto cswap = [&](int i, int j) { if (comp(data[j], data[i])) std::swap(data[i], data[j]); };
            cswap(0, 2); cswap(1, 3); cswap(0, 1); cswap(2, 4);
            cswap(1, 2); cswap(3, 4); cswap(0, 1); cswap(2, 3); cswap(1, 2);
        }

        // ============================================================
        // Insertion sort variants
        // ============================================================
        // 前提：data[low - 1] 必须是整个 [low, high] 区间的最小值（哨兵）。
        // 调用方（三路划分 / block_partition_right）保证父区间的 pivot 在
        // data[low - 1] 位置且小于等于区间内所有元素。
        // 如果将来修改划分逻辑，务必维持这个不变量，否则 while 循环会越界。
        template <typename T, typename Compare>
        void unguarded_insertion_sort(std::vector<T>& data, int low, int high, Compare comp)
        {
            for (int i = low + 1; i <= high; ++i)
            {
                T key = std::move(data[i]);
                int j = i - 1;
                while (comp(key, data[j])) { data[j + 1] = std::move(data[j]); --j; }
                data[j + 1] = std::move(key);
            }
        }

        template <typename T, typename Compare>
        bool partial_insertion_sort(std::vector<T>& data, int low, int high, Compare comp, int limit = PARTIAL_INSERTION_LIMIT)
        {
            if (low >= high) return true;
            int moves = 0;
            for (int i = low + 1; i <= high; ++i)
            {
                T key = data[i];
                int j = i - 1;
                while (j >= low && comp(key, data[j]))
                {
                    data[j + 1] = data[j];
                    --j;
                    if (++moves > limit) return false;
                }
                data[j + 1] = key;
            }
            return true;
        }

        // Linear insertion sort (faster than binary for n < 64)
        template <typename T, typename Compare>
        void linear_insertion_sort(std::vector<T>& data, int low, int high, Compare comp)
        {
            for (int i = low + 1; i <= high; ++i)
            {
                T key = std::move(data[i]);
                int j = i - 1;
                while (j >= low && comp(key, data[j]))
                {
                    data[j + 1] = std::move(data[j]);
                    --j;
                }
                data[j + 1] = std::move(key);
            }
        }

        template <typename T, typename Compare>
        void insertion_sort(std::vector<T>& data, int low, int high, Compare comp)
        {
            for (int i = low + 1; i <= high; ++i)
            {
                T key = std::move(data[i]);
                int left = low, right = i;
                while (left < right)
                {
                    int mid = left + (right - left) / 2;
                    if (comp(key, data[mid])) right = mid;
                    else left = mid + 1;
                }
                for (int j = i; j > left; --j) data[j] = std::move(data[j - 1]);
                data[left] = std::move(key);
            }
        }

        // ============================================================
        // Median of three
        // ============================================================
        template <typename T, typename Compare>
        inline const T& median_of_three(const T& a, const T& b, const T& c, Compare comp)
        {
            if (comp(b, a))
            {
                if (comp(c, b)) return c;
                if (comp(c, a)) return a;
                return b;
            }
            else
            {
                if (comp(c, a)) return a;
                if (comp(c, b)) return c;
                return b;
            }
        }

        template <typename T, typename Compare>
        inline void median_swap3(std::vector<T>& data, int a, int b, int c, Compare comp)
        {
            if (comp(data[b], data[a])) std::swap(data[a], data[b]);
            if (comp(data[c], data[b])) std::swap(data[b], data[c]);
            if (comp(data[b], data[a])) std::swap(data[a], data[b]);
        }

        // ============================================================
        // Block Partition (BlockQuicksort-inspired, two-way)
        // ============================================================
        template <typename T, typename Compare>
        inline int block_partition_right(std::vector<T>& data, int low, int high, Compare comp)
        {
            constexpr int BLOCK = 64;

            T pivot = std::move(data[low]);

            int first = low;
            int last = high + 1;

            while (comp(data[++first], pivot)) {}
            if (first - 1 == low)
                while (first < last && !comp(data[--last], pivot)) {}
            else
                while (!comp(data[--last], pivot)) {}

            if (first >= last)
            {
                data[low] = std::move(data[last]);
                data[last] = std::move(pivot);
                return last;
            }

            std::swap(data[first], data[last]);
            ++first;

            alignas(64) unsigned char offsets_l[BLOCK + 64];
            alignas(64) unsigned char offsets_r[BLOCK + 64];

            int l_base = first;
            int r_base = last;
            int l_cnt = 0, r_cnt = 0;
            int l_start = 0, r_start = 0;

            while (first < last)
            {
                int avail = last - first;
                int left_take = (l_cnt == 0) ? (r_cnt == 0 ? avail / 2 : avail) : 0;
                int right_take = (r_cnt == 0) ? (avail - left_take) : 0;

                if (left_take >= BLOCK)
                {
                    for (int i = 0; i < BLOCK; )
                    {
                        offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                        offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                        offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                        offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                        offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                        offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                        offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                        offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                    }
                }
                else
                {
                    for (int i = 0; i < left_take; )
                    {
                        offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                    }
                }

                if (right_take >= BLOCK)
                {
                    for (int i = 0; i < BLOCK; )
                    {
                        offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                        offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                        offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                        offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                        offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                        offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                        offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                        offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                    }
                }
                else
                {
                    for (int i = 0; i < right_take; )
                    {
                        offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                    }
                }

                int n = (l_cnt < r_cnt) ? l_cnt : r_cnt;
                for (int i = 0; i < n; ++i)
                    std::swap(data[l_base + offsets_l[l_start + i]], data[r_base - offsets_r[r_start + i]]);
                l_cnt -= n; r_cnt -= n;
                l_start += n; r_start += n;

                if (l_cnt == 0) { l_start = 0; l_base = first; }
                if (r_cnt == 0) { r_start = 0; r_base = last; }
            }

            if (l_cnt)
            {
                for (int i = 0; i < l_cnt; ++i)
                    std::swap(data[l_base + offsets_l[l_start + i]], data[--last]);
                first = last;
            }
            if (r_cnt)
            {
                for (int i = 0; i < r_cnt; ++i)
                    std::swap(data[r_base - offsets_r[r_start + i]], data[first++]);
                last = first;
            }

            int pivot_pos = first - 1;
            data[low] = std::move(data[pivot_pos]);
            data[pivot_pos] = std::move(pivot);
            return pivot_pos;
        }

        // ============================================================
        // Bit reinterpretation
        // ============================================================
        template <typename T>
        inline unsigned int to_unsigned_32(T x)
        {
            static_assert(std::is_same_v<T, int> || std::is_same_v<T, unsigned int>, "bad type");
            if constexpr (std::is_same_v<T, int>)
                return static_cast<unsigned int>(x) ^ (1u << 31);
            else
                return x;
        }

        template <typename T>
        inline unsigned long long to_unsigned_64(T x)
        {
            static_assert(std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long>, "bad type");
            if constexpr (std::is_same_v<T, long long>)
                return static_cast<unsigned long long>(x) ^ (1ull << 63);
            else
                return x;
        }

        // ============================================================
        // Counting sort with bitmap-skip + stack bucket
        // ============================================================
        template <typename T>
        void counting_sort_range_vec(std::vector<T>& data, T minV, unsigned int range, bool descending = false)
        {
            const int n = static_cast<int>(data.size());
            const int SIZE = static_cast<int>(range);
            const unsigned int umin = static_cast<unsigned int>(minV);

            std::vector<int> cnt(SIZE, 0);
            for (int i = 0; i < n; ++i)
                ++cnt[static_cast<unsigned int>(data[i]) - umin];

            const int WORDS = (SIZE + 63) >> 6;
            std::vector<unsigned long long> bitmap(WORDS, 0ull);
            int nonEmpty = 0;
            for (int b = 0; b < SIZE; ++b)
            {
                if (cnt[b])
                {
                    bitmap[b >> 6] |= 1ull << (b & 63);
                    ++nonEmpty;
                }
            }

            std::vector<T> buffer(n);
            int pos = 0;

            if (!descending)
            {
                if (nonEmpty * 4 > SIZE * 3)
                {
                    for (int b = 0; b < SIZE; ++b)
                    {
                        T val = static_cast<T>(umin + static_cast<unsigned int>(b));
                        int c = cnt[b];
                        while (c--) buffer[pos++] = val;
                    }
                }
                else
                {
                    for (int w = 0; w < WORDS; ++w)
                    {
                        unsigned long long word = bitmap[w];
                        while (word)
                        {
                            int bit = ctz64(word);
                            int b = (w << 6) | bit;
                            T val = static_cast<T>(umin + static_cast<unsigned int>(b));
                            int c = cnt[b];
                            while (c--) buffer[pos++] = val;
                            word &= word - 1;
                        }
                    }
                }
            }
            else
            {
                if (nonEmpty * 4 > SIZE * 3)
                {
                    for (int b = SIZE - 1; b >= 0; --b)
                    {
                        T val = static_cast<T>(umin + static_cast<unsigned int>(b));
                        int c = cnt[b];
                        while (c--) buffer[pos++] = val;
                    }
                }
                else
                {
                    for (int w = WORDS - 1; w >= 0; --w)
                    {
                        unsigned long long word = bitmap[w];
                        while (word)
                        {
                            int bit = 63 - clz64(word);
                            int b = (w << 6) | bit;
                            T val = static_cast<T>(umin + static_cast<unsigned int>(b));
                            int c = cnt[b];
                            while (c--) buffer[pos++] = val;
                            word &= ~(1ull << bit);
                        }
                    }
                }
            }

            data.swap(buffer);
        }

        template <typename T>
        void counting_sort_range_stack(std::vector<T>& data, T minV, unsigned int range, bool descending = false)
        {
            const int n = static_cast<int>(data.size());
            const int SIZE = static_cast<int>(range);
            const unsigned int umin = static_cast<unsigned int>(minV);

            int cnt[COUNTING_STACK_BUCKETS];
            for (int i = 0; i < SIZE; ++i) cnt[i] = 0;

            for (int i = 0; i < n; ++i)
                ++cnt[static_cast<unsigned int>(data[i]) - umin];

            std::vector<T> buffer(n);
            int pos = 0;

            if (!descending)
            {
                for (int b = 0; b < SIZE; ++b)
                {
                    T val = static_cast<T>(umin + static_cast<unsigned int>(b));
                    int c = cnt[b];
                    while (c--) buffer[pos++] = val;
                }
            }
            else
            {
                for (int b = SIZE - 1; b >= 0; --b)
                {
                    T val = static_cast<T>(umin + static_cast<unsigned int>(b));
                    int c = cnt[b];
                    while (c--) buffer[pos++] = val;
                }
            }

            data.swap(buffer);
        }

        template <typename T>
        bool try_counting_sort(std::vector<T>& data, bool descending = false)
        {
            if constexpr (!(std::is_same_v<T, int> || std::is_same_v<T, unsigned int>))
                return false;
            else
            {
                const int n = static_cast<int>(data.size());
                if (n <= 1) return true;

                T minV = data[0], maxV = data[0];
                for (int i = 1; i < n; ++i)
                {
                    if (data[i] < minV) minV = data[i];
                    if (data[i] > maxV) maxV = data[i];
                }

                unsigned long long rangeLL =
                    static_cast<unsigned long long>(maxV) -
                    static_cast<unsigned long long>(minV) + 1ull;

                if (rangeLL > COUNTING_SORT_RANGE_LIMIT) return false;
                if (rangeLL > static_cast<unsigned long long>(n) * 8ull) return false;

                unsigned int range = static_cast<unsigned int>(rangeLL);
                if (range <= COUNTING_STACK_BUCKETS)
                    counting_sort_range_stack(data, minV, range, descending);
                else
                    counting_sort_range_vec(data, minV, range, descending);
                return true;
            }
        }

        // ============================================================
        // 11-bit single-threaded radix sort (3 passes)
        // Direction handled by reversing pos array initialization.
        // ============================================================
        template <typename T>
        bool radix_sort_32_11bit(std::vector<T>& data, bool descending = false)
        {
            const int n = static_cast<int>(data.size());
            if (n <= 1) return true;
            if constexpr (!(std::is_same_v<T, int> || std::is_same_v<T, unsigned int>))
                return false;

            const bool do_prefetch = (n > PREFETCH_THRESHOLD);
            std::vector<T> buffer(n);

            constexpr int SHIFT1 = 0, SHIFT2 = 11, SHIFT3 = 22;
            constexpr int MASK1 = 0x7FF, MASK2 = 0x7FF, MASK3 = 0x3FF;
            constexpr int SIZE1 = 2048, SIZE2 = 2048, SIZE3 = 1024;

            {
                int count[SIZE1] = { 0 };
                for (int i = 0; i < n; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < n) RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                    ++count[(to_unsigned_32(data[i]) >> SHIFT1) & MASK1];
                }
                int pos[SIZE1];
                if (!descending)
                {
                    int sum = 0;
                    for (int i = 0; i < SIZE1; ++i) { pos[i] = sum; sum += count[i]; }
                }
                else
                {
                    pos[SIZE1 - 1] = 0;
                    for (int i = SIZE1 - 2; i >= 0; --i) pos[i] = pos[i + 1] + count[i + 1];
                }
                for (int i = 0; i < n; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < n) RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                    int idx = (to_unsigned_32(data[i]) >> SHIFT1) & MASK1;
                    buffer[pos[idx]++] = data[i];
                }
                data.swap(buffer);
            }
            {
                int count[SIZE2] = { 0 };
                for (int i = 0; i < n; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < n) RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                    ++count[(to_unsigned_32(data[i]) >> SHIFT2) & MASK2];
                }
                int pos[SIZE2];
                if (!descending)
                {
                    int sum = 0;
                    for (int i = 0; i < SIZE2; ++i) { pos[i] = sum; sum += count[i]; }
                }
                else
                {
                    pos[SIZE2 - 1] = 0;
                    for (int i = SIZE2 - 2; i >= 0; --i) pos[i] = pos[i + 1] + count[i + 1];
                }
                for (int i = 0; i < n; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < n) RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                    int idx = (to_unsigned_32(data[i]) >> SHIFT2) & MASK2;
                    buffer[pos[idx]++] = data[i];
                }
                data.swap(buffer);
            }
            {
                int count[SIZE3] = { 0 };
                for (int i = 0; i < n; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < n) RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                    ++count[(to_unsigned_32(data[i]) >> SHIFT3) & MASK3];
                }
                int pos[SIZE3];
                if (!descending)
                {
                    int sum = 0;
                    for (int i = 0; i < SIZE3; ++i) { pos[i] = sum; sum += count[i]; }
                }
                else
                {
                    pos[SIZE3 - 1] = 0;
                    for (int i = SIZE3 - 2; i >= 0; --i) pos[i] = pos[i + 1] + count[i + 1];
                }
                for (int i = 0; i < n; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < n) RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                    int idx = (to_unsigned_32(data[i]) >> SHIFT3) & MASK3;
                    buffer[pos[idx]++] = data[i];
                }
                data.swap(buffer);
            }
            return true;
        }

        // ============================================================
        // MSD+LSD hybrid radix sort
        // Both MSD buckets and LSD inner sort respect descending.
        // ============================================================
        template <typename T>
        void radix_sort_bucket_24bit(std::vector<T>& data, std::vector<T>& buffer, int lo, int hi)
        {
            const int m = hi - lo;
            if (m <= 1) return;

            const bool do_prefetch = (m > PREFETCH_THRESHOLD);

            {
                int count[LSD12_SIZE] = { 0 };
                for (int i = 0; i < m; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < m) RIPPLE_PREFETCH(&data[lo + i + PREFETCH_DISTANCE]);
                    ++count[to_unsigned_32(data[lo + i]) & LSD12_MASK];
                }
                int pos[LSD12_SIZE];
                {
                    int sum = 0;
                    for (int i = 0; i < LSD12_SIZE; ++i) { pos[i] = sum; sum += count[i]; }
                }
                for (int i = 0; i < m; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < m) RIPPLE_PREFETCH(&data[lo + i + PREFETCH_DISTANCE]);
                    int idx = to_unsigned_32(data[lo + i]) & LSD12_MASK;
                    buffer[pos[idx]++] = data[lo + i];
                }
            }

            {
                int count[LSD12_SIZE] = { 0 };
                for (int i = 0; i < m; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < m) RIPPLE_PREFETCH(&buffer[i + PREFETCH_DISTANCE]);
                    ++count[(to_unsigned_32(buffer[i]) >> 12) & LSD12_MASK];
                }
                int pos[LSD12_SIZE];
                {
                    int sum = 0;
                    for (int i = 0; i < LSD12_SIZE; ++i) { pos[i] = sum; sum += count[i]; }
                }
                for (int i = 0; i < m; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < m) RIPPLE_PREFETCH(&buffer[i + PREFETCH_DISTANCE]);
                    int idx = (to_unsigned_32(buffer[i]) >> 12) & LSD12_MASK;
                    data[lo + pos[idx]++] = buffer[i];
                }
            }
        }

        template <typename T>
        bool radix_sort_32_msd_lsd(std::vector<T>& data, bool descending = false)
        {
            const int n = static_cast<int>(data.size());
            if (n <= 1) return true;
            if constexpr (!(std::is_same_v<T, int> || std::is_same_v<T, unsigned int>))
                return false;

            const bool do_prefetch = (n > PREFETCH_THRESHOLD);

            std::vector<T> msdBuffer(n);
            int count[MSD_BUCKET_SIZE] = { 0 };

            for (int i = 0; i < n; ++i)
            {
                if (do_prefetch && i + PREFETCH_DISTANCE < n) RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                ++count[(to_unsigned_32(data[i]) >> MSD_SHIFT) & MSD_BUCKET_MASK];
            }

            int pos[MSD_BUCKET_SIZE];
            {
                int sum = 0;
                for (int i = 0; i < MSD_BUCKET_SIZE; ++i) { pos[i] = sum; sum += count[i]; }
            }

            for (int i = 0; i < n; ++i)
            {
                if (do_prefetch && i + PREFETCH_DISTANCE < n) RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                int idx = (to_unsigned_32(data[i]) >> MSD_SHIFT) & MSD_BUCKET_MASK;
                msdBuffer[pos[idx]++] = data[i];
            }
            data.swap(msdBuffer);

            // Reuse msdBuffer as LSD scratch space. After the swap above,
            // msdBuffer contains the old (unsorted) data, which is no longer
            // needed. It can be safely used as scratch space.
            std::vector<T>& lsdBuffer = msdBuffer;

            int bucketStart = 0;
            for (int b = 0; b < MSD_BUCKET_SIZE; ++b)
            {
                int bucketEnd = bucketStart + count[b];
                if (count[b] <= 16) {
                    // 全局用 reverse 统一处理降序，所以桶内一律升序
                    insertion_sort(data, bucketStart, bucketEnd - 1, std::less<T>{});
                }
                else if (count[b] > 1) {
                    radix_sort_bucket_24bit(data, lsdBuffer, bucketStart, bucketEnd);
                }
                bucketStart = bucketEnd;
            }

            if (descending) std::reverse(data.begin(), data.end());

            return true;
        }

        // ============================================================
        // Multi-threaded 8-bit radix sort
        // Direction handled by final reverse.
        // ============================================================
        template <typename T>
        bool radix_sort_32_parallel(std::vector<T>& data, bool descending = false)
        {
            const int n = static_cast<int>(data.size());
            if (n <= 1) return true;
            if constexpr (!(std::is_same_v<T, int> || std::is_same_v<T, unsigned int>))
                return false;

            unsigned int numThreads = std::thread::hardware_concurrency();
            if (numThreads == 0) numThreads = 4;
            if (numThreads > 16) numThreads = 16;
            unsigned int maxByData = static_cast<unsigned int>(n / 500000);
            if (maxByData < 1) maxByData = 1;
            if (numThreads > maxByData) numThreads = maxByData;

            if (numThreads <= 1) return radix_sort_32_11bit(data, descending);

            const bool do_prefetch = (n > PREFETCH_THRESHOLD);

            struct alignas(64) AlignedBucket {
                std::array<int, 256> data;
            };

            std::vector<T> buffer(n);
            const int blockSize = (n + static_cast<int>(numThreads) - 1) / static_cast<int>(numThreads);

            // 持久状态：复用，避免每趟重建
            std::vector<AlignedBucket> localCount(numThreads);
            std::vector<AlignedBucket> threadPos(numThreads);
            std::array<int, 256> bucketStart;
            std::vector<std::thread> threads;
            threads.reserve(numThreads - 1);

            for (int pass = 0; pass < 4; ++pass)
            {
                const int shift = pass * 8;
                const int mask = 0xFF;

                for (unsigned int t = 0; t < numThreads; ++t)
                    localCount[t].data.fill(0);
                threads.clear();

                // ==================== 计数阶段 ====================
                for (unsigned int t = 1; t < numThreads; ++t)
                {
                    threads.emplace_back([&, t]() {
                        int start = static_cast<int>(t) * blockSize;
                        int end = std::min(start + blockSize, n);
                        auto& lc = localCount[t].data;
                        for (int i = start; i < end; ++i)
                        {
                            if (do_prefetch && i + PREFETCH_DISTANCE < end)
                                RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                            ++lc[(to_unsigned_32(data[i]) >> shift) & mask];
                        }
                        });
                }
                {
                    int end = std::min(blockSize, n);
                    auto& lc = localCount[0].data;
                    for (int i = 0; i < end; ++i)
                    {
                        if (do_prefetch && i + PREFETCH_DISTANCE < end)
                            RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                        ++lc[(to_unsigned_32(data[i]) >> shift) & mask];
                    }
                }
                for (auto& th : threads) th.join();

                // ==================== 前缀和 ====================
                {
                    int sum = 0;
                    for (int b = 0; b < 256; ++b)
                    {
                        bucketStart[b] = sum;
                        int total = 0;
                        for (unsigned int t = 0; t < numThreads; ++t)
                            total += localCount[t].data[b];
                        sum += total;
                    }
                }
                for (unsigned int t = 0; t < numThreads; ++t)
                {
                    for (int b = 0; b < 256; ++b)
                    {
                        int offset = bucketStart[b];
                        for (unsigned int u = 0; u < t; ++u)
                            offset += localCount[u].data[b];
                        threadPos[t].data[b] = offset;
                    }
                }

                // ==================== 分散阶段 ====================
                threads.clear();
                for (unsigned int t = 1; t < numThreads; ++t)
                {
                    threads.emplace_back([&, t]() {
                        int start = static_cast<int>(t) * blockSize;
                        int end = std::min(start + blockSize, n);
                        auto pos = threadPos[t].data;
                        for (int i = start; i < end; ++i)
                        {
                            if (do_prefetch && i + PREFETCH_DISTANCE < end)
                                RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                            int idx = (to_unsigned_32(data[i]) >> shift) & mask;
                            buffer[pos[idx]++] = data[i];
                        }
                        });
                }
                {
                    int end = std::min(blockSize, n);
                    auto pos = threadPos[0].data;
                    for (int i = 0; i < end; ++i)
                    {
                        if (do_prefetch && i + PREFETCH_DISTANCE < end)
                            RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                        int idx = (to_unsigned_32(data[i]) >> shift) & mask;
                        buffer[pos[idx]++] = data[i];
                    }
                }
                for (auto& th : threads) th.join();

                data.swap(buffer);
            }

            if (descending) std::reverse(data.begin(), data.end());

            return true;
        }

        // ============================================================
        // 64-bit radix sort (direction handled via pos initialization)
        // ============================================================
        template <typename T>
        inline unsigned int radix_byte_64(T x, int byte)
        {
            return (to_unsigned_64(x) >> (byte * 8)) & 0xFF;
        }

        template <typename T>
        bool radix_sort_64(std::vector<T>& data, bool descending = false)
        {
            const int n = static_cast<int>(data.size());
            if (n <= 1) return true;
            if constexpr (!(std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long>))
                return false;

            const bool do_prefetch = (n > PREFETCH_THRESHOLD);

            std::vector<T> buffer(n);
            int count[256];
            int pos[256];

            for (int byte = 0; byte < 8; ++byte)
            {
                std::memset(count, 0, sizeof(count));
                for (int i = 0; i < n; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < n) RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                    ++count[radix_byte_64(data[i], byte)];
                }
                if (!descending)
                {
                    int sum = 0;
                    for (int i = 0; i < 256; ++i) { pos[i] = sum; sum += count[i]; }
                }
                else
                {
                    pos[255] = 0;
                    for (int i = 254; i >= 0; --i) pos[i] = pos[i + 1] + count[i + 1];
                }
                for (int i = 0; i < n; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < n) RIPPLE_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                    int idx = radix_byte_64(data[i], byte);
                    buffer[pos[idx]++] = std::move(data[i]);
                }
                data.swap(buffer);
            }
            return true;
        }

        // ============================================================
        // Radix dispatcher
        // ============================================================
        template <typename T>
        bool dispatch_radix(std::vector<T>& data, bool descending)
        {
            if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int>)
            {
                const int n = static_cast<int>(data.size());
                if (n >= PARALLEL_RADIX_MIN)
                    return radix_sort_32_parallel(data, descending);
                else if (n >= MSD_LSD_THRESHOLD)
                    return radix_sort_32_msd_lsd(data, descending);
                else
                    return radix_sort_32_11bit(data, descending);
            }
            else if constexpr (std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long>)
            {
                return radix_sort_64(data, descending);
            }
            else
            {
                (void)descending;
                return false;
            }
        }

        // ============================================================
        // Heap sort helpers
        // ============================================================
        template <typename T, typename Compare>
        void sift_down(std::vector<T>& data, int root, int end, int offset, Compare comp)
        {
            T root_val = std::move(data[root]);
            while (true)
            {
                int child = (root - offset) * 2 + 1 + offset;
                if (child > end) break;
                if (child + 1 <= end && comp(data[child], data[child + 1])) child++;
                if (!comp(root_val, data[child])) break;
                data[root] = std::move(data[child]);
                root = child;
            }
            data[root] = std::move(root_val);
        }

        template <typename T, typename Compare>
        void heap_sort(std::vector<T>& data, int low, int high, Compare comp)
        {
            int n = high - low + 1;
            for (int start = low + n / 2 - 1; start >= low; --start) sift_down(data, start, high, low, comp);
            for (int end = high; end > low; --end)
            {
                std::swap(data[low], data[end]);
                sift_down(data, low, end - 1, low, comp);
            }
        }

        // ============================================================
        // Small-data specialized sorter (n < SMALL_DATA_THRESHOLD)
        // Design: pdqsort-inspired, no radix, no prescan, linear insertion
        // ============================================================
        template <typename T, typename Compare>
        void small_quicksort(std::vector<T>& data, int low, int high, Compare comp, int depth)
        {
            while (low < high)
            {
                int len = high - low + 1;

                if (len <= SMALL_INSERTION_THRESHOLD)
                {
                    if (len <= SORT_NETWORK_THRESHOLD)
                    {
                        switch (len)
                        {
                        case 1: break;
                        case 2: if (comp(data[high], data[low])) std::swap(data[low], data[high]); break;
                        case 3: sort_network_3(data.data() + low, comp); break;
                        case 4: sort_network_4(data.data() + low, comp); break;
                        case 5: sort_network_5(data.data() + low, comp); break;
                        default: linear_insertion_sort(data, low, high, comp); break;
                        }
                    }
                    else
                    {
                        linear_insertion_sort(data, low, high, comp);
                    }
                    return;
                }

                if (depth > MAX_SMALL_DEPTH)
                {
                    heap_sort(data, low, high, comp);
                    return;
                }

                int mid = low + (high - low) / 2;
                median_swap3(data, low, mid, high, comp);
                std::swap(data[low], data[mid]);
                T pivot = data[low];

                // Correct three-way partition:
                // lt starts at low, i starts at low+1, gt starts at high.
                // pivot stays at data[low] until the very end.
                int lt = low;
                int i = low + 1;
                int gt = high;

                while (i <= gt)
                {
                    if (comp(data[i], pivot))
                    {
                        ++lt;
                        std::swap(data[lt], data[i]);
                        ++i;
                    }
                    else if (comp(pivot, data[i]))
                    {
                        std::swap(data[i], data[gt]);
                        --gt;
                    }
                    else
                    {
                        ++i;
                    }
                }
                std::swap(data[low], data[lt]);

                int l_size = lt - low;
                int r_size = high - gt;

                if (l_size <= 16 && r_size <= 16)
                {
                    bool left_ok = partial_insertion_sort(data, low, lt - 1, comp);
                    bool right_ok = partial_insertion_sort(data, gt + 1, high, comp);
                    if (left_ok && right_ok) return;
                }

                if (l_size < r_size)
                {
                    small_quicksort(data, low, lt - 1, comp, depth + 1);
                    low = gt + 1;
                }
                else
                {
                    small_quicksort(data, gt + 1, high, comp, depth + 1);
                    high = lt - 1;
                }
                depth++;
            }
        }

        template <typename T, typename Compare>
        void small_sort_core(std::vector<T>& data, int low, int high, Compare comp)
        {
            int len = high - low + 1;
            if (len <= 1) return;

            if (len <= SMALL_INSERTION_THRESHOLD)
            {
                if (len <= SORT_NETWORK_THRESHOLD)
                {
                    switch (len)
                    {
                    case 1: break;
                    case 2: if (comp(data[high], data[low])) std::swap(data[low], data[high]); break;
                    case 3: sort_network_3(data.data() + low, comp); break;
                    case 4: sort_network_4(data.data() + low, comp); break;
                    case 5: sort_network_5(data.data() + low, comp); break;
                    default: linear_insertion_sort(data, low, high, comp); break;
                    }
                }
                else
                {
                    linear_insertion_sort(data, low, high, comp);
                }
                return;
            }

            small_quicksort(data, low, high, comp, 0);
        }

        // ============================================================
        // Pre-scan
        // ============================================================
        enum ScanClass {
            SCAN_UNKNOWN,
            SCAN_ALL_SAME,
            SCAN_SORTED,
            SCAN_REVERSED,
            SCAN_NEARLY_SORTED
        };

        template <typename T, typename Compare>
        ScanClass try_prescan(std::vector<T>& data, int n, Compare comp)
        {
            if (n > QUICK_REJECT_SAMPLE)
            {
                int descCount = 0;
                int ascCount = 0;
                for (int i = 1; i < QUICK_REJECT_SAMPLE; ++i)
                {
                    if (comp(data[i], data[i - 1])) ++descCount;
                    else if (comp(data[i - 1], data[i])) ++ascCount;
                }
                bool monotonicDesc = (descCount >= QUICK_REJECT_SAMPLE - 2);
                bool monotonicAsc = (ascCount >= QUICK_REJECT_SAMPLE - 2);
                if (!monotonicDesc && !monotonicAsc && descCount > 8 && ascCount > 8)
                    return SCAN_UNKNOWN;
            }

            bool allSame = true;
            bool isSorted = true;
            bool isReversed = true;
            int invCount = 0;

            for (int i = 1; i < n; ++i)
            {
                bool lt = comp(data[i], data[i - 1]);
                bool gt = comp(data[i - 1], data[i]);
                if (lt) { isSorted = false; ++invCount; }
                if (gt) { isReversed = false; }
                if (lt || gt) { allSame = false; }
            }

            if (allSame) return SCAN_ALL_SAME;
            if (isSorted) return SCAN_SORTED;
            if (isReversed) return SCAN_REVERSED;

            if (invCount < MAX_NEARLY_SORTED_INVERSION && n > 32)
            {
                // 注意：整数类型不在这里做 nearly-sorted 判定。
                // prescan 只在 n >= PRESCAN_THRESHOLD (2048) 时调用，
                // 整数类型由 radix 路径兜底更快。
                if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
                    std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long>)
                    return SCAN_UNKNOWN;
                else
                    return SCAN_NEARLY_SORTED;
            }

            return SCAN_UNKNOWN;
        }

        // ============================================================
        // Introsort core
        // ============================================================
        struct Range { int low; int high; int depth; int bad_allowed; };

        template <typename T, typename Compare>
        void intro_sort_core(std::vector<T>& data, int k, Compare comp)
        {
            int n = static_cast<int>(data.size());
            if (n <= 1) return;

            // Small-data fast path: skip prescan and radix sort entirely.
            // For n < =SMALL_DATA_THRESHOLD, the small_sort_core is faster
            // than the general prescan + radix path.
            if (k == -1 && n <= SMALL_DATA_THRESHOLD)
            {
                small_sort_core(data, 0, n - 1, comp);
                return;
            }

            constexpr bool is_default_compare = std::is_same_v<Compare, std::less<T>>;
            constexpr bool is_descending = std::is_same_v<Compare, std::greater<T>>;
            constexpr bool can_use_prescan = is_default_compare || is_descending;
            constexpr bool can_use_radix = is_default_compare || is_descending;

            if (k == -1 && can_use_prescan && n >= PRESCAN_THRESHOLD)
            {
                ScanClass sc = try_prescan(data, n, comp);
                if (sc == SCAN_ALL_SAME || sc == SCAN_SORTED) return;
                if (sc == SCAN_REVERSED)
                {
                    std::reverse(data.begin(), data.end());
                    return;
                }
                if (sc == SCAN_NEARLY_SORTED)
                {
                    insertion_sort(data, 0, n - 1, comp);
                    return;
                }
            }

            if (k == -1 && can_use_radix && n > RADIX_SORT_THRESHOLD && n < 500000000)
            {
                if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int>)
                {
                    if (try_counting_sort(data, is_descending)) return;
                    if (dispatch_radix(data, is_descending)) return;
                }
                if constexpr (std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long>)
                {
                    if (dispatch_radix(data, is_descending)) return;
                }
            }

            if (k != -1 && k > 0 && k <= n)
            {
                if (k == n) {}
                else if (k <= n / 10 && n >= 100)
                {
                    std::vector<T> heap(data.begin(), data.begin() + k);
                    std::make_heap(heap.begin(), heap.end(), comp);
                    for (int i = k; i < n; ++i)
                    {
                        if (comp(data[i], heap[0]))
                        {
                            std::pop_heap(heap.begin(), heap.end(), comp);
                            heap.back() = data[i];
                            std::push_heap(heap.begin(), heap.end(), comp);
                        }
                    }
                    for (int i = 0; i < k; ++i) data[i] = std::move(heap[i]);
                    insertion_sort(data, 0, k - 1, comp);
                    return;
                }
                else if (k <= n / 2)
                {
                    std::partial_sort(data.begin(), data.begin() + k, data.end(), comp);
                    return;
                }
                else
                {
                    std::nth_element(data.begin(), data.begin() + k, data.end(), comp);
                    std::sort(data.begin(), data.begin() + k, comp);
                    return;
                }
            }

            int max_depth = 0;
            while ((1LL << max_depth) < n) max_depth++;
            max_depth *= 2;

            std::vector<Range> stk;
            stk.reserve(static_cast<size_t>(max_depth) + 16);
            stk.push_back({ 0, n - 1, max_depth, std::max(1, max_depth / 4) });

            while (!stk.empty())
            {
                Range r = stk.back();
                stk.pop_back();
                int low = r.low;
                int high = r.high;
                int depth = r.depth;
                int bad_allowed = r.bad_allowed;

                while (true)
                {
                    int len = high - low + 1;

                    if (len < INSERTION_SORT_THRESHOLD)
                    {
                        if (len <= SORT_NETWORK_THRESHOLD)
                        {
                            switch (len)
                            {
                            case 1: break;
                            case 2: if (comp(data[high], data[low])) std::swap(data[low], data[high]); break;
                            case 3: sort_network_3(data.data() + low, comp); break;
                            case 4: sort_network_4(data.data() + low, comp); break;
                            case 5: sort_network_5(data.data() + low, comp); break;
                            default: insertion_sort(data, low, high, comp); break;
                            }
                        }
                        else if (low == 0)
                            insertion_sort(data, low, high, comp);
                        else
                            unguarded_insertion_sort(data, low, high, comp);
                        break;
                    }

                    if (depth == 0)
                    {
                        heap_sort(data, low, high, comp);
                        break;
                    }

                    int mid = low + (high - low) / 2;
                    T pivot;

                    if (len >= NINTHER_THRESHOLD)
                    {
                        median_swap3(data, low, low + 1, low + 2, comp);
                        median_swap3(data, mid - 1, mid, mid + 1, comp);
                        median_swap3(data, high - 2, high - 1, high, comp);
                        median_swap3(data, low + 1, mid, high - 1, comp);
                        std::swap(data[low], data[mid]);
                        pivot = data[low];
                    }
                    else
                    {
                        median_swap3(data, low, mid, high, comp);
                        std::swap(data[low], data[mid]);
                        pivot = data[low];
                    }

                    int lt, gt;

                    if (len >= 512 && is_default_compare)
                    {
                        int pivot_pos = block_partition_right(data, low, high, comp);
                        lt = pivot_pos;
                        gt = pivot_pos;
                    }
                    else
                    {
                        int i = low;
                        lt = low;
                        gt = high;
                        while (i <= gt)
                        {
                            if (comp(data[i], pivot))
                            {
                                std::swap(data[lt], data[i]);
                                lt++;
                                i++;
                            }
                            else if (comp(pivot, data[i]))
                            {
                                std::swap(data[i], data[gt]);
                                gt--;
                            }
                            else
                                i++;
                        }
                    }

                    int l_size = lt - low;
                    int r_size = high - gt;
                    bool highly_unbalanced = (l_size < len / 8 || r_size < len / 8);

                    if (highly_unbalanced)
                    {
                        if (--bad_allowed == 0)
                        {
                            heap_sort(data, low, high, comp);
                            break;
                        }
                        if (l_size >= 4)
                        {
                            std::swap(data[low], data[low + l_size / 4]);
                            std::swap(data[lt - 1], data[lt - 1 - l_size / 4]);
                        }
                        if (r_size >= 4)
                        {
                            std::swap(data[gt + 1], data[gt + 1 + r_size / 4]);
                            std::swap(data[high], data[high - r_size / 4]);
                        }
                    }
                    else
                    {
                        if (l_size <= 16 && r_size <= 16)
                        {
                            bool left_ok = partial_insertion_sort(data, low, lt - 1, comp);
                            bool right_ok = partial_insertion_sort(data, gt + 1, high, comp);
                            if (left_ok && right_ok) break;
                        }
                    }

                    if (l_size < r_size)
                    {
                        if (gt + 1 <= high) stk.push_back({ gt + 1, high, depth - 1, bad_allowed });
                        high = lt - 1;
                    }
                    else
                    {
                        if (low <= lt - 1) stk.push_back({ low, lt - 1, depth - 1, bad_allowed });
                        low = gt + 1;
                    }
                    depth--;
                }
            }
        }

    }  // namespace detail

    // ============================================================
    // Public API
    // ============================================================

    template <typename T, typename Compare = std::less<T>>
    void sort(std::vector<T>& data, Compare comp = {})
    {
        detail::intro_sort_core(data, -1, comp);
    }

    template <typename Iter, typename Compare = std::less<typename std::iterator_traits<Iter>::value_type>>
    void sort(Iter begin, Iter end, Compare comp = {})
    {
        using Cat = typename std::iterator_traits<Iter>::iterator_category;
        static_assert(
            std::is_base_of_v<std::random_access_iterator_tag, Cat>,
            "ripple_sort::sort requires random-access iterators (use std::vector or std::array)");

        using T = typename std::iterator_traits<Iter>::value_type;
        std::vector<T> data(begin, end);
        detail::intro_sort_core(data, -1, comp);
        std::copy(data.begin(), data.end(), begin);
    }

    template <typename T, std::size_t N, typename Compare = std::less<T>>
    void sort(T(&arr)[N], Compare comp = {})
    {
        std::vector<T> data(arr, arr + N);
        detail::intro_sort_core(data, -1, comp);
        std::copy(data.begin(), data.end(), arr);
    }

    template <typename T, typename Compare = std::less<T>>
    void topk(std::vector<T>& data, int k, Compare comp = {})
    {
        if (k <= 0) return;
        int n = static_cast<int>(data.size());
        if (k > n) k = n;
        if (k == n) { sort(data, comp); return; }
        detail::intro_sort_core(data, k, comp);
    }

    template <typename T, typename Compare = std::less<T>>
    void radix_sort(std::vector<T>& data, Compare comp = {})
    {
        constexpr bool is_desc = std::is_same_v<Compare, std::greater<T>>;
        detail::dispatch_radix(data, is_desc);
    }

    // ============================================================
    // LazySortedView — zero-cost top-K view
    // ============================================================
    template <typename T, typename Compare = std::less<T>>
    class LazySortedView
    {
    public:
        LazySortedView(const std::vector<T>& source, int k, Compare comp = {})
            : m_source(source)
            , m_k((k < 0) ? 0 : (k > (int)source.size() ? (int)source.size() : k))
            , m_comp(comp)
            , m_ready(false)
        {
        }

        const T& operator[](int idx) const
        {
            ensure_sorted();
            return m_sorted[idx];
        }

        const T* begin() const { ensure_sorted(); return m_sorted.data(); }
        const T* end() const { ensure_sorted(); return m_sorted.data() + m_k; }

        int size() const { return m_k; }

        void materialize() const { ensure_sorted(); }

    private:
        void ensure_sorted() const
        {
            if (m_ready) return;
            m_sorted = m_source;
            ripple_sort::topk(m_sorted, m_k, m_comp);
            m_ready = true;
        }

        const std::vector<T>& m_source;
        mutable std::vector<T> m_sorted;
        int m_k;
        Compare m_comp;
        mutable bool m_ready;
    };

}  // namespace ripple_sort
