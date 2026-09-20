# Ripple Sort

> A single-header adaptive sorting library for C++17/20.
> Prescan → counting / MSD+LSD / 11-bit / parallel radix → introsort fallback.

**Current version:** v4.3.8 (Correctness & Cleanup)
**Tests:** 40/40 PASS
**Performance:** 1M ints — 5.7× faster than `std::sort`, 2.4× faster than `pdqsort`
**Dependencies:** none.

---

## What it is

A drop-in sorting library for integer types. One header, no build system, no dependencies.

```cpp
#include "ripple_sort.h"

std::vector<int> data = { /* ... */ };

ripple_sort::sort(data);                        // ascending
ripple_sort::sort(data, std::greater<int>{});   // descending
ripple_sort::topk(data, 10);                    // only top-K smallest
Five-tier adaptive dispatch
Automatically picks the fastest path based on input size and data shape:

Small-data fast path (n ≤ 256) — pdqsort-style, skips prescan and radix entirely

Prescan (n ≥ 2048) — single pass detects all-equal / already-sorted / fully-reversed / nearly-sorted

Counting sort — when value range is compact (range ≤ 8n and ≤ 2²⁰), uses bitmap-skip counting

Radix sort — three variants by size:

11-bit × 3-pass LSD (n < 50K)

MSD 8-bit + LSD 12-bit hybrid (50K ≤ n < 2M)

8-bit × 4-pass multi-threaded (n ≥ 2M)

Introsort fallback — quicksort + heapsort + insertion sort, guarantees O(n log n)

Everything else
Single header — ripple_sort.h, zero dependencies

Full ascending / descending support on every path

Top-K lazy sort — only sorts the first K elements

LazySortedView — zero-cost top-K view

Software prefetch — threshold-gated on radix read loops

Cross-platform — MSVC (x86 / x64 / ARM64), GCC, Clang

C++17 and C++20 compatible

Performance
Test machine: i5-4210U (2C/4T, 2013) + single-channel DDR3 + 4GB RAM
Compiler: clang-cl / C++20 / O2 / AVX2 / Ot
Method: median of 64 runs

1M random ints
Algorithm	Median	Relative to Ripple
Ripple	20.85 ms	1.00×
pdqsort	50.00 ms	2.40× slower
std::sort	118.30 ms	5.68× slower
std::stable_sort	106.05 ms	5.09× slower
100K random ints
Algorithm	Median	Relative to Ripple
Ripple	1.71 ms	1.00×
pdqsort	4.26 ms	2.49× slower
std::sort	9.52 ms	5.57× slower
std::stable_sort	8.47 ms	4.96× slower
Full test suite: 40 cases — boundaries, threshold edges, structured patterns, random data, extreme distributions. All PASS.

Where the name comes from
This library started as a Python ripple prototype:

python
def ripple_sort(arr):
    data = arr.copy()
    n = len(data)
    i = 0
    while i < n - 1:
        if data[i] > data[i + 1]:
            j = i
            while j < n - 1 and data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
                j += 1
            if i > 0: i -= 1
            else: i = 0
        else:
            i += 1
    return data
When an inversion is found, it propagates rightward, then steps back — like a ripple spreading outward.

It grew into a C++ hybrid sorting library across three designers:

First generation — introsort skeleton + prescan + radix + prefetch

Second generation — full descending support + small-data path + MSD+LSD hybrid

Third generation — bug fixes + the "three-ratio criterion" + two cross-dimensional experiments (in-place radix, PCF Learned Sort — both disproved, archived in experiments/)

The name never changed. It's still about pushing outward, one ring at a time.

Known limitations
Only integer types (int, unsigned int, long long, unsigned long long)

Allocates an O(n) scratch buffer on radix paths (not externalized)

In-place radix sort and learned sort were tried and disproved on this hardware — see experiments/

Repository layout
text
ripple_sort/
├── ripple_sort.h              ← the library (single header)
├── test_compare.cpp           ← benchmark / correctness harness
├── pdqsort.h                  ← reference baseline
├── baseline_v4.3.8.txt        ← current performance baseline
└── experiments/               ← archived cross-dimensional attempts
    ├── README.md
    ├── ripple_sort_inplace.h
    └── ripple_learned.h
How to build the tests
text
clang-cl test_compare.cpp /std:c++20 /O2 /arch:AVX2 /Ot /EHsc /MD
Then run the produced binary. It prints a 40-case comparison against pdqsort, std::sort, and std::stable_sort.

License
[YOUR LICENSE HERE]

Author
[YOUR NAME / CONTACT HERE]

text

---

## Short description for GitHub "About" field
Single-header adaptive sorting library for C++17/20.
1M ints: 5.7× faster than std::sort, 2.4× faster than pdqsort.
40/40 tests pass. Zero dependencies. Drop-in.

text

*(152 chars — fits GitHub's 350-char limit easily.)*

---

## If you want a one-liner for social media

> Released Ripple Sort v4.3.8 — a single-header C++ sorting library.
> 1M ints: 5.7× faster than std::sort, 2.4× faster than pdqsort.
> Five-tier adaptive dispatch, 40/40 tests pass, zero dependencies.
> Grew out of a Python ripple prototype across three designers.
>  [repo link]

---

## Notes

- **License placeholder** — pick MIT / BSD-2 / Apache-2.0 / whatever you want and fill it in.
- **Author placeholder** — put your name / GitHub handle there.
- If you don't want the "three generations" story in the README, cut the `Where the name comes from` section. But I'd keep it — **projects with a story get more stars than projects without one.**

**Copy, paste, fill in the two placeholders. You're done.** 
