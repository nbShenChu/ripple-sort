// ============================================================
// ripple_sort.hpp - Ripple-Sort v2.1.1 Adaptive
// Single-header, zero-dependency sorting library.
// ============================================================
// Features:
//   - Introsort with 3-way partition
//   - 32/64-bit radix sort for integers and floats
//   - Block partition for large arrays
//   - Run detection for partially sorted data
//   - NaN-safe for floating point types
//   - Top-K support (heap select / partial_sort / nth_element)
//   - In-place and copy-out interfaces
// ============================================================
// Usage:
//   #include "ripple_sort.hpp"
//   std::vector<int> data = ...;
//   ripple_sort_inplace(data);
//   auto sorted = ripple_sort(data);
//   auto top5 = ripple_sort_topk(data, 5);
// ============================================================
// License: MIT
// ============================================================

#ifndef RIPPLE_SORT_HPP
#define RIPPLE_SORT_HPP

#include <vector>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <cstring>
#include <limits>
#include <cmath>

namespace ripple {

// ============================================================
// Configuration constants
// ============================================================
constexpr int INSERTION_SORT_THRESHOLD = 64;
constexpr int NINTHER_THRESHOLD = 128;
constexpr int MAX_NEARLY_SORTED_INVERSION = 4;
constexpr int NEARLY_SORTED_SAMPLE = 8;
constexpr int PRESCAN_THRESHOLD = 128;
constexpr int PARTIAL_INSERTION_LIMIT = 8;
constexpr int RADIX_SORT_THRESHOLD = 256;
constexpr int SORT_NETWORK_THRESHOLD = 5;
constexpr int BLOCK_SIZE = 64;

// ============================================================
// Internal helpers
// ============================================================
namespace detail {

// Sorting networks (3/4/5 elements)
template<typename T, typename Compare>
inline void sort_network_3(T* data, Compare comp) {
    auto cswap = [&](int i, int j) {
        if (comp(data[j], data[i])) std::swap(data[i], data[j]);
    };
    cswap(0, 2);
    cswap(0, 1);
    cswap(1, 2);
}

template<typename T, typename Compare>
inline void sort_network_4(T* data, Compare comp) {
    auto cswap = [&](int i, int j) {
        if (comp(data[j], data[i])) std::swap(data[i], data[j]);
    };
    cswap(0, 2);
    cswap(1, 3);
    cswap(0, 1);
    cswap(2, 3);
    cswap(1, 2);
}

template<typename T, typename Compare>
inline void sort_network_5(T* data, Compare comp) {
    auto cswap = [&](int i, int j) {
        if (comp(data[j], data[i])) std::swap(data[i], data[j]);
    };
    cswap(0, 2);
    cswap(1, 3);
    cswap(0, 1);
    cswap(2, 4);
    cswap(1, 2);
    cswap(3, 4);
    cswap(0, 1);
    cswap(2, 3);
    cswap(1, 2);
}

// Unguarded insertion sort (assumes sentinel at low-1)
template<typename T, typename Compare>
inline void unguarded_insertion_sort(std::vector<T>& data, int low, int high, Compare comp) {
    for (int i = low + 1; i <= high; ++i) {
        T key = std::move(data[i]);
        int j = i - 1;
        while (comp(key, data[j])) {
            data[j + 1] = std::move(data[j]);
            --j;
        }
        data[j + 1] = std::move(key);
    }
}

// Copy-free median of three
template<typename T, typename Compare>
inline const T& median_of_three(const T& a, const T& b, const T& c, Compare comp) {
    if (comp(b, a)) {
        if (comp(c, b)) return c;
        if (comp(c, a)) return a;
        return b;
    } else {
        if (comp(c, a)) return a;
        if (comp(c, b)) return c;
        return b;
    }
}

// 32-bit radix sort
template<typename T>
inline unsigned int to_unsigned_32(T x) {
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, unsigned int> || std::is_same_v<T, float>,
                  "to_unsigned_32 only supports 32-bit types");
    if constexpr (std::is_same_v<T, int>) {
        return static_cast<unsigned int>(x) ^ (1u << 31);
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        return x;
    } else if constexpr (std::is_same_v<T, float>) {
        unsigned int u;
        std::memcpy(&u, &x, sizeof(float));
        if (std::isnan(x)) return 0xFFFFFFFFu;
        unsigned int sign_bit = static_cast<unsigned int>(-static_cast<int>(u >> 31));
        return u ^ (sign_bit | 0x80000000u);
    }
}

template<typename T>
inline unsigned int radix_byte_32(T x, int byte) {
    return (to_unsigned_32(x) >> (byte * 8)) & 0xFF;
}

template<typename T>
inline bool radix_sort_32(std::vector<T>& data) {
    const int n = static_cast<int>(data.size());
    if (n <= 1) return true;
    if constexpr (!std::is_arithmetic_v<T>) return false;
    if constexpr (!(std::is_same_v<T, int> || std::is_same_v<T, unsigned int> || std::is_same_v<T, float>)) {
        return false;
    }

    std::vector<T> buffer(n);
    int count[256], pos[256];
    for (int byte = 0; byte < 4; ++byte) {
        std::memset(count, 0, sizeof(count));
        for (int i = 0; i < n; ++i) ++count[radix_byte_32(data[i], byte)];
        pos[0] = 0;
        for (int i = 1; i < 256; ++i) pos[i] = pos[i - 1] + count[i - 1];
        for (int i = 0; i < n; ++i) {
            int idx = radix_byte_32(data[i], byte);
            buffer[pos[idx]++] = std::move(data[i]);
        }
        data.swap(buffer);
    }
    return true;
}

// 64-bit radix sort
template<typename T>
inline unsigned long long to_unsigned_64(T x) {
    static_assert(std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long> ||
                  std::is_same_v<T, double>,
                  "to_unsigned_64 only supports 64-bit types");
    if constexpr (std::is_same_v<T, long long>) {
        return static_cast<unsigned long long>(x) ^ (1ull << 63);
    } else if constexpr (std::is_same_v<T, unsigned long long>) {
        return x;
    } else if constexpr (std::is_same_v<T, double>) {
        unsigned long long u;
        std::memcpy(&u, &x, sizeof(double));
        if (std::isnan(x)) return 0xFFFFFFFFFFFFFFFFull;
        unsigned long long sign_bit = static_cast<unsigned long long>(-static_cast<long long>(u >> 63));
        return u ^ (sign_bit | 0x8000000000000000ull);
    }
}

template<typename T>
inline unsigned int radix_byte_64(T x, int byte) {
    return (to_unsigned_64(x) >> (byte * 8)) & 0xFF;
}

template<typename T>
inline bool radix_sort_64(std::vector<T>& data) {
    const int n = static_cast<int>(data.size());
    if (n <= 1) return true;
    if constexpr (!std::is_arithmetic_v<T>) return false;
    if constexpr (!(std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long> ||
                    std::is_same_v<T, double>)) {
        return false;
    }

    std::vector<T> buffer(n);
    int count[256], pos[256];
    for (int byte = 0; byte < 8; ++byte) {
        std::memset(count, 0, sizeof(count));
        for (int i = 0; i < n; ++i) ++count[radix_byte_64(data[i], byte)];
        pos[0] = 0;
        for (int i = 1; i < 256; ++i) pos[i] = pos[i - 1] + count[i - 1];
        for (int i = 0; i < n; ++i) {
            int idx = radix_byte_64(data[i], byte);
            buffer[pos[idx]++] = std::move(data[i]);
        }
        data.swap(buffer);
    }
    return true;
}

// Block 3-way partition
template<typename T, typename Compare>
struct BlockThreeWayPartition {
    static constexpr int BLOCK_SIZE = 64;
    int l_offsets[BLOCK_SIZE];
    int r_offsets[BLOCK_SIZE];
    int l_cnt, r_cnt;

    BlockThreeWayPartition() : l_cnt(0), r_cnt(0) {
        std::fill_n(l_offsets, BLOCK_SIZE, 0);
        std::fill_n(r_offsets, BLOCK_SIZE, 0);
    }

    void reset() { l_cnt = r_cnt = 0; }

    void flush_left(std::vector<T>& data, int& lt, int& i) {
        for (int k = 0; k < l_cnt; ++k) {
            std::swap(data[lt++], data[i]);
            i++;
        }
        l_cnt = 0;
    }

    void flush_right(std::vector<T>& data, int& gt, int& i) {
        for (int k = 0; k < r_cnt; ++k) {
            std::swap(data[i], data[gt--]);
        }
        r_cnt = 0;
    }
};

template<typename T, typename Compare>
inline void block_three_way_partition(std::vector<T>& data, int low, int high, int& lt, int& gt,
                                       T pivot, Compare comp) {
    const int n = high - low + 1;
    if (n < 128) {
        lt = low;
        int i = low;
        gt = high;
        while (i <= gt) {
            if (comp(data[i], pivot)) {
                std::swap(data[lt], data[i]);
                lt++;
                i++;
            } else if (comp(pivot, data[i])) {
                std::swap(data[i], data[gt]);
                gt--;
            } else {
                i++;
            }
        }
        return;
    }

    lt = low;
    int i = low;
    gt = high;
    BlockThreeWayPartition<T, Compare> block;

    while (i <= gt) {
        block.reset();
        const int remaining = gt - i + 1;
        const int batch = std::min(BlockThreeWayPartition<T, Compare>::BLOCK_SIZE, remaining);
        int j = i;

        for (int k = 0; k < batch; ++k, ++j) {
            if (comp(data[j], pivot)) {
                block.l_offsets[block.l_cnt++] = j - i;
            } else if (comp(pivot, data[j])) {
                block.r_offsets[block.r_cnt++] = j - i;
            }
        }

        if (block.l_cnt > 0 && block.r_cnt > 0) {
            int cnt = std::min(block.l_cnt, block.r_cnt);
            for (int k = 0; k < cnt; ++k) {
                std::swap(data[i + block.l_offsets[k]], data[i + block.r_offsets[k]]);
            }
            for (int k = 0; k < block.l_cnt; ++k) {
                std::swap(data[lt], data[i + block.l_offsets[k]]);
                lt++;
            }
            for (int k = 0; k < block.r_cnt; ++k) {
                std::swap(data[i + block.r_offsets[k]], data[gt]);
                gt--;
            }
            i += batch;
        } else if (block.l_cnt > 0) {
            for (int k = 0; k < block.l_cnt; ++k) {
                std::swap(data[lt], data[i + block.l_offsets[k]]);
                lt++;
            }
            i += block.l_cnt;
        } else if (block.r_cnt > 0) {
            for (int k = 0; k < block.r_cnt; ++k) {
                std::swap(data[i + block.r_offsets[k]], data[gt]);
                gt--;
            }
            i += block.r_cnt;
        } else {
            i += batch;
        }
    }
}

// Run detection
struct Run {
    int start;
    int len;
    bool descending;
};

template<typename T, typename Compare>
inline std::vector<Run> detect_runs(std::vector<T>& data, Compare comp) {
    std::vector<Run> runs;
    const int n = static_cast<int>(data.size());
    if (n <= 1) return runs;

    int i = 0;
    while (i < n) {
        int start = i;
        if (i + 1 >= n) {
            runs.push_back({start, 1, false});
            break;
        }
        bool desc = comp(data[i + 1], data[i]);
        while (i + 1 < n && (comp(data[i + 1], data[i]) == desc ||
               (!comp(data[i], data[i + 1]) && !comp(data[i + 1], data[i])))) {
            i++;
        }
        int len = i - start + 1;
        if (desc) {
            std::reverse(data.begin() + start, data.begin() + i + 1);
        }
        runs.push_back({start, len, desc});
        i++;
    }
    return runs;
}

template<typename T, typename Compare>
inline void merge_runs(std::vector<T>& data, const std::vector<Run>& runs, Compare comp) {
    if (runs.size() <= 1) return;
    std::vector<Run> current = runs;
    while (current.size() > 1) {
        std::vector<Run> next;
        for (size_t i = 0; i < current.size(); i += 2) {
            if (i + 1 < current.size()) {
                const Run& left = current[i];
                const Run& right = current[i + 1];
                std::inplace_merge(data.begin() + left.start,
                                   data.begin() + left.start + left.len,
                                   data.begin() + left.start + left.len + right.len,
                                   comp);
                next.push_back({left.start, left.len + right.len, false});
            } else {
                next.push_back(current[i]);
            }
        }
        current = std::move(next);
    }
}

// Partial insertion sort with move limit
template<typename T, typename Compare>
inline bool partial_insertion_sort(std::vector<T>& data, int low, int high, Compare comp,
                                    int limit = PARTIAL_INSERTION_LIMIT) {
    if (low >= high) return true;
    int moves = 0;
    for (int i = low + 1; i <= high; ++i) {
        T key = data[i];
        int j = i - 1;
        while (j >= low && comp(key, data[j])) {
            data[j + 1] = data[j];
            --j;
            if (++moves > limit) return false;
        }
        data[j + 1] = key;
    }
    return true;
}

// Standard binary insertion sort
template<typename T, typename Compare>
inline void insertion_sort(std::vector<T>& data, int low, int high, Compare comp) {
    for (int i = low + 1; i <= high; ++i) {
        T key = std::move(data[i]);
        int left = low, right = i;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (comp(key, data[mid])) right = mid;
            else left = mid + 1;
        }
        for (int j = i; j > left; --j) data[j] = std::move(data[j - 1]);
        data[left] = std::move(key);
    }
}

// Heap sort helpers
template<typename T, typename Compare>
inline void sift_down(std::vector<T>& data, int root, int end, int offset, Compare comp) {
    T root_val = std::move(data[root]);
    while (true) {
        int child = (root - offset) * 2 + 1 + offset;
        if (child > end) break;
        if (child + 1 <= end && comp(data[child], data[child + 1])) child++;
        if (!comp(root_val, data[child])) break;
        data[root] = std::move(data[child]);
        root = child;
    }
    data[root] = std::move(root_val);
}

template<typename T, typename Compare>
inline void heap_sort(std::vector<T>& data, int low, int high, Compare comp) {
    int n = high - low + 1;
    for (int start = low + n / 2 - 1; start >= low; --start)
        sift_down(data, start, high, low, comp);
    for (int end = high; end > low; --end) {
        std::swap(data[low], data[end]);
        sift_down(data, low, end - 1, low, comp);
    }
}

// ============================================================
// Core sorting engine
// ============================================================
struct Range {
    int low, high, depth, bad_allowed;
};

template<typename T, typename Compare>
inline void intro_sort_core(std::vector<T>& data, int k, Compare comp) {
    int n = static_cast<int>(data.size());
    if (n <= 1) return;

    bool topk_processed = false;

    // Radix sort fast path
    if (k == -1 && n > RADIX_SORT_THRESHOLD && n < 10000000) {
        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int> || std::is_same_v<T, float>) {
            if (radix_sort_32(data)) return;
        }
        if constexpr (std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long> ||
                    std::is_same_v<T, double>) {
            if (radix_sort_64(data)) return;
        }
    }

    // Pre-scan
    if (n >= PRESCAN_THRESHOLD && (k == -1 || k > 16)) {
        // All equal
        if (!comp(data[0], data[n - 1]) && !comp(data[n - 1], data[0])) {
            bool all_same = true;
            for (int i = 1; i < n; ++i) {
                if (comp(data[i], data[0]) || comp(data[0], data[i])) {
                    all_same = false;
                    break;
                }
            }
            if (all_same) return;
        }

        // Sorted detection
        bool sample_sorted = true;
        int sample_points[] = {0, n / 4, n / 2, 3 * n / 4};
        for (int i = 0; i < 3; ++i) {
            if (comp(data[sample_points[i + 1]], data[sample_points[i]])) {
                sample_sorted = false;
                break;
            }
        }
        if (sample_sorted) {
            bool full_sorted = true;
            for (int i = 0; i < n - 1; ++i) {
                if (comp(data[i + 1], data[i])) {
                    full_sorted = false;
                    break;
                }
            }
            if (full_sorted) return;
        }

        // Three-probe inversion count
        int inv = 0;
        if (n >= 2 && comp(data[1], data[0])) ++inv;
        if (n >= 4 && comp(data[n / 2 + 1], data[n / 2])) ++inv;
        if (n >= 3 && comp(data[n - 1], data[n - 2])) ++inv;

        // Run detection
        if (inv <= 2 && n > 256) {
            auto runs = detect_runs(data, comp);
            if (runs.size() > 1 && runs.size() <= 4) {
                merge_runs(data, runs, comp);
                return;
            }
        }

        // Nearly sorted
        if (inv <= 1) {
            int inversions = 0;
            int sample_sz = std::min(NEARLY_SORTED_SAMPLE, n / 3);
            if (sample_sz > 0) {
                int seg_offsets[] = {0, n / 2 - sample_sz / 2, n - sample_sz};
                for (int off : seg_offsets) {
                    int pos = off;
                    if (pos < 0) pos = 0;
                    if (pos + sample_sz > n) pos = n - sample_sz;
                    for (int i = pos; i < pos + sample_sz - 1 && i < n - 1; ++i) {
                        if (comp(data[i + 1], data[i])) {
                            if (++inversions >= MAX_NEARLY_SORTED_INVERSION) break;
                        }
                    }
                    if (inversions >= MAX_NEARLY_SORTED_INVERSION) break;
                }
            }
            if (inversions < MAX_NEARLY_SORTED_INVERSION) {
                insertion_sort(data, 0, n - 1, comp);
                return;
            }
        }

        // Full reverse
        if (inv >= 3) {
            bool full_reverse = true;
            for (int i = 0; i < n - 1; ++i) {
                if (!comp(data[i + 1], data[i])) {
                    full_reverse = false;
                    break;
                }
            }
            if (full_reverse) {
                std::reverse(data.begin(), data.end());
                return;
            }
        }
    }

    // Top-K
    if (k != -1 && k > 0 && k <= n) {
        if (k == n) {
            // full sort
        } else if (k <= n / 10 && n >= 100) {
            topk_processed = true;
            std::vector<T> heap(data.begin(), data.begin() + k);
            std::make_heap(heap.begin(), heap.end(), comp);
            for (int i = k; i < n; ++i) {
                if (comp(data[i], heap[0])) {
                    std::pop_heap(heap.begin(), heap.end(), comp);
                    heap.back() = data[i];
                    std::push_heap(heap.begin(), heap.end(), comp);
                }
            }
            for (int i = 0; i < k; ++i) data[i] = std::move(heap[i]);
            insertion_sort(data, 0, k - 1, comp);
            return;
        } else if (k <= n / 2) {
            topk_processed = true;
            std::partial_sort(data.begin(), data.begin() + k, data.end(), comp);
            return;
        } else {
            topk_processed = true;
            std::nth_element(data.begin(), data.begin() + k, data.end(), comp);
            std::sort(data.begin(), data.begin() + k, comp);
            return;
        }
    }

    // Introsort
    int max_depth = 0;
    while ((1LL << max_depth) < n) max_depth++;
    max_depth *= 2;

    std::vector<Range> stk;
    stk.reserve(max_depth + 16);
    int initial_bad_allowed = std::max(1, max_depth / 4);
    stk.push_back({0, n - 1, max_depth, initial_bad_allowed});

    while (!stk.empty()) {
        Range r = stk.back();
        stk.pop_back();
        int low = r.low;
        int high = r.high;
        int depth = r.depth;
        int bad_allowed = r.bad_allowed;

        while (true) {
            int len = high - low + 1;

            if (len < INSERTION_SORT_THRESHOLD) {
                if (len <= SORT_NETWORK_THRESHOLD) {
                    switch (len) {
                        case 1: break;
                        case 2: if (comp(data[high], data[low])) std::swap(data[low], data[high]); break;
                        case 3: sort_network_3(data.data() + low, comp); break;
                        case 4: sort_network_4(data.data() + low, comp); break;
                        case 5: sort_network_5(data.data() + low, comp); break;
                        default: insertion_sort(data, low, high, comp); break;
                    }
                } else if (low == 0) {
                    insertion_sort(data, low, high, comp);
                } else {
                    unguarded_insertion_sort(data, low, high, comp);
                }
                break;
            }

            if (depth == 0) {
                heap_sort(data, low, high, comp);
                break;
            }

            int mid = low + (high - low) / 2;
            T pivot;
            if (len >= NINTHER_THRESHOLD) {
                const T& m1 = median_of_three(data[low], data[low + 1], data[low + 2], comp);
                const T& m2 = median_of_three(data[mid - 1], data[mid], data[mid + 1], comp);
                const T& m3 = median_of_three(data[high - 2], data[high - 1], data[high], comp);
                pivot = median_of_three(m1, m2, m3, comp);
            } else {
                if (comp(data[mid], data[low])) std::swap(data[mid], data[low]);
                if (comp(data[high], data[low])) std::swap(data[high], data[low]);
                if (comp(data[high], data[mid])) std::swap(data[high], data[mid]);
                pivot = std::move(data[mid]);
            }

            // NaN handling
            if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(pivot)) {
                    int nan_count = 0;
                    for (int i = low; i <= high; ++i) {
                        if (std::isnan(data[i])) {
                            std::swap(data[i], data[high - nan_count]);
                            nan_count++;
                        }
                    }
                    high -= nan_count;
                    if (low >= high) break;
                    continue;
                }
            }

            int lt, gt;
            if (len > 256) {
                block_three_way_partition(data, low, high, lt, gt, pivot, comp);
            } else {
                lt = low;
                int i = low;
                gt = high;
                while (i <= gt) {
                    if (comp(data[i], pivot)) {
                        std::swap(data[lt], data[i]);
                        lt++;
                        i++;
                    } else if (comp(pivot, data[i])) {
                        std::swap(data[i], data[gt]);
                        gt--;
                    } else {
                        i++;
                    }
                }
            }

            int l_size = lt - low;
            int r_size = high - gt;
            bool highly_unbalanced = (l_size < len / 8 || r_size < len / 8);

            if (highly_unbalanced) {
                if (--bad_allowed == 0) {
                    heap_sort(data, low, high, comp);
                    break;
                }
                if (l_size >= 4) {
                    std::swap(data[low], data[low + l_size / 4]);
                    std::swap(data[lt - 1], data[lt - 1 - l_size / 4]);
                }
                if (r_size >= 4) {
                    std::swap(data[gt + 1], data[gt + 1 + r_size / 4]);
                    std::swap(data[high], data[high - r_size / 4]);
                }
            } else {
                if (l_size <= 16 && r_size <= 16) {
                    bool left_ok = partial_insertion_sort(data, low, lt - 1, comp);
                    bool right_ok = partial_insertion_sort(data, gt + 1, high, comp);
                    if (left_ok && right_ok) break;
                }
            }

            if (l_size < r_size) {
                if (gt + 1 <= high) stk.push_back({gt + 1, high, depth - 1, bad_allowed});
                high = lt - 1;
            } else {
                if (low <= lt - 1) stk.push_back({low, lt - 1, depth - 1, bad_allowed});
                low = gt + 1;
            }
            depth--;
        }
    }

    if (k != -1 && k > 0 && k <= n && !topk_processed && k <= n / 2) {
        insertion_sort(data, 0, k - 1, comp);
    }
}

} // namespace detail

// ============================================================
// Public interfaces
// ============================================================

// Copy-out sort (returns a new sorted vector)
template<typename T, typename Compare = std::less<T>>
inline std::vector<T> sort(const std::vector<T>& arr, Compare comp = {}) {
    std::vector<T> data = arr;
    detail::intro_sort_core(data, -1, comp);
    return data;
}

// In-place sort (modifies original vector)
template<typename T, typename Compare = std::less<T>>
inline void sort_inplace(std::vector<T>& data, Compare comp = {}) {
    detail::intro_sort_core(data, -1, comp);
}

// Top-K: returns the k smallest elements (copy-out)
template<typename T, typename Compare = std::less<T>>
inline std::vector<T> sort_topk(const std::vector<T>& arr, int k, Compare comp = {}) {
    std::vector<T> data = arr;
    if (k <= 0) return {};
    int n = static_cast<int>(data.size());
    if (k > n) k = n;
    detail::intro_sort_core(data, k, comp);
    data.resize(k);
    return data;
}

// Top-K: in-place version (modifies original vector, only first k are sorted)
template<typename T, typename Compare = std::less<T>>
inline void sort_topk_inplace(std::vector<T>& data, int k, Compare comp = {}) {
    if (k <= 0) return;
    int n = static_cast<int>(data.size());
    if (k > n) k = n;
    detail::intro_sort_core(data, k, comp);
}

} // namespace ripple

#endif // RIPPLE_SORT_HPP
