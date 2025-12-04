# Histogram Triton Implementation Report

## Code Structure
- We flatten the `[length, num_channels]` tensor into a 1‑D buffer so each Triton “program” (i.e., block) handles `BLOCK_SIZE` independent elements.
- The grid is 1‑D: `pid = tl.program_id(0)` selects the chunk, `offsets = pid * BLOCK_SIZE + tl.arange(...)` enumerates the elements handled by that program, and a mask guards out-of-bounds threads.
- Each thread loads one byte, decodes its channel via `offset % num_channels`, uses the sample value itself as the bin index, and issues a `tl.atomic_add` into a `[num_channels, num_bins]` histogram laid out row-major.
- There is no tiling across channels/bins yet; all parallelism comes from independent atomics over the flattened stream, so threads contend whenever multiple samples hit the same `(channel, bin)` pair.

## Performance
- **Runtime (not yet collected):** This kernel requires a CUDA-capable GPU with Triton; the current dev host does not expose one, so no empirical runtime has been recorded yet. The next action item is to invoke `python problems/eval.py benchmark problems/histogram/test_cases/test.txt` (or submit via `popcorn-cli`) on an H100 node to log an actual number.

## Additional Statistics Observed
- Static occupancy-style numbers: with `BLOCK_SIZE=1024` and ~5.37e8 elements per test, the launch grid is ~5.25e5 programs; every program instantiates 32 warps, so we expect high theoretical occupancy but heavy global-atomic pressure.
- Memory-traffic estimate: each element reads 1 byte and performs an atomic add to a 4-byte counter, so the best-case bandwidth demand is ≳2.5 GB per run; this sets a roofline indicating the kernel will likely be memory/atomic bound.

## Conclusions from the (Limited) Measurements
- Even without runtime, the traffic estimate and atomic-per-element structure indicate the kernel will bottleneck on global atomic throughput rather than arithmetic or shared-memory bandwidth.
- The flattened launch pattern ignores spatial locality, so hot channels/bins will cause multiple warps across the grid to serialize on the same cache lines, further depressing SM utilization.

## Performance-Limiting Hypothesis
- The primary limiter is global-atomic contention: each of the ~5.37e8 updates targets only 512×256 counters, so per-bin pressure is high (~4K updates per bin on average, with some bins far hotter). This hypothesis comes from counting the ratio of updates to bins plus the lack of privatization in the kernel.

## Next Steps Suggested by the Hypothesis
- Introduce per-block privatized histograms in shared memory (e.g., tile a subset of bins/channels per thread block), accumulate locally, then flush once per block to reduce atomic frequency.
- Alternatively, split the tensor by channels so that each kernel instance handles a subset of channels exclusively, eliminating modulo arithmetic and reducing contention on shared rows.
- Tune `BLOCK_SIZE`/number of warps per program to balance latency hiding with per-program register pressure once the privatization strategy is in place, and benchmark again on the H100 queue.

---


