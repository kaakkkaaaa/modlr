*Updated November 2026 for Mojo 1.2.0. The original post (31 Aug 2025) used Mojo syntax that no longer compiles. Every snippet below has been checked against Mojo 1.2.0, and the benchmark numbers come from a real run. A summary of what changed is at the end.*

Single Instruction, Multiple Data (SIMD) is one of the most powerful optimization techniques available to modern programmers. Mojo gives us zero-cost abstractions and direct hardware access, so we can use SIMD fully and still keep the code readable. Today we'll explore SIMD through a practical example: efficient matrix operations.

## What is SIMD?

SIMD lets your CPU perform the same operation on several data elements at once. Instead of processing one floating-point number at a time, modern processors can handle 4, 8, or even 16 values in a single instruction. For suitable workloads this often gives a 4x to 16x speedup.

The key insight is that many computational problems are embarrassingly parallel at the data level. Matrix operations, image processing, and numerical computations all have this property, which makes them ideal candidates for SIMD.

## SIMD in Action: Simple Examples

Let's start with a concrete example. Imagine you want to add two arrays of numbers:

```
Array A: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Array B: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
Result:  [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5]
```

### The Scalar Approach (Traditional)

Without SIMD, your CPU processes one pair at a time:

```mojo
def add_arrays_scalar(
    a: ImmPointer[Float32, _],
    b: ImmPointer[Float32, _],
    result: MutPointer[Float32, _],
    size: Int,
):
    for i in range(size):
        result[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]
```

A few Mojo 1.2 details are already visible here:

- `UnsafePointer` is gone. We use `ImmPointer` for read-only inputs and `MutPointer` for the output.
- The `_` lets the compiler infer the pointer's *origin*, meaning where its memory came from.
- Indexing a pointer is now spelled `p[unsafe_offset=i]`, which makes the unchecked access obvious when you read the code.

**CPU Instructions:**

- Load a[0], load b[0] → add → store result[0]
- Load a[1], load b[1] → add → store result[1]
- Load a[2], load b[2] → add → store result[2]
- ... (8 separate iterations for 8 elements)

### The SIMD Approach (Vectorized)

With SIMD, your CPU processes several elements at the same time:

```mojo
from std.algorithm import vectorize

def add_arrays_simd(
    a: ImmPointer[Float32, _],
    b: ImmPointer[Float32, _],
    result: MutPointer[Float32, _],
    size: Int,
):
    comptime simd_width = 8  # Process 8 float32s at once

    def vectorized_add[width: Int](i: Int) {imm a, imm b, imm result}:
        var vec_a = a.unsafe_offset(i).unsafe_load[width=width]()  # Load 8 from A
        var vec_b = b.unsafe_offset(i).unsafe_load[width=width]()  # Load 8 from B
        var vec_result = vec_a + vec_b                             # Add all 8 pairs!
        result.unsafe_offset(i).unsafe_store(vec_result)           # Store 8 results

    vectorize[simd_width](size, vectorized_add)
```

Things worth noticing:

- **`comptime`** replaces `alias` for compile-time constants.
- **Closures are plain nested `def`s with a capture list** (`{imm a, imm b, imm result}`). The old `@parameter fn` decorator is gone, and the capture list states exactly what the closure borrows.
- **`vectorize` takes the closure as a normal argument**: `vectorize[simd_width](size, closure)`. It calls the closure with `width = simd_width` for the main body. Any leftover elements at the end are handled automatically.
- **Loads and stores** are `unsafe_load[width=...]()` and `unsafe_store(...)` on an offset pointer.

**CPU Instructions (with AVX2):**

- Load 8 floats from A, load 8 floats from B → add all 8 pairs → store 8 results

**Result: 8x fewer instructions!**

### Visual Representation

Here's what happens inside the CPU:

```
Scalar Processing (8 iterations):
Step 1: [1.0] + [0.5] = [1.5]
Step 2: [2.0] + [1.5] = [3.5]
Step 3: [3.0] + [2.5] = [5.5]
... (continues for each element)

SIMD Processing (1 iteration):
Step 1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] +
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5] =
        [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5]
```

### A More Practical Example: Scaling Values

Here's another common pattern, multiplying an array by a constant:

```mojo
from std.sys import simd_width_of

def scale_array_naive(data: MutPointer[Float32, _], size: Int, factor: Float32):
    for i in range(size):
        data[unsafe_offset=i] = data[unsafe_offset=i] * factor


def scale_array_simd(data: MutPointer[Float32, _], size: Int, factor: Float32):
    comptime simd_width = simd_width_of[DType.float32]()

    def vectorized_scale[width: Int](i: Int) {imm data, imm factor}:
        var vec_data = data.unsafe_offset(i).unsafe_load[width=width]()
        var vec_factor = SIMD[DType.float32, width](factor)  # Broadcast scalar to vector
        data.unsafe_offset(i).unsafe_store(vec_data * vec_factor)

    vectorize[simd_width](size, vectorized_scale)
```

Here the scalar `factor` gets **broadcast** to every lane of the SIMD vector, and then all the multiplications happen at once. Note the rename: `simdwidthof` is now `simd_width_of`, imported from `std.sys`.

### When SIMD Works Best

SIMD shines when you have:

- **Regular data patterns** (arrays, matrices)
- **The same operation on many data elements** (the "Single Instruction" part)
- **Enough data parallelism** (at least 4–16 elements to process)
- **Straight-line arithmetic** rather than branch-heavy code

Now that we understand the fundamentals, let's apply them to matrix operations.

## Building Our Foundation: The Matrix Structure

Here's a basic matrix implementation in Mojo 1.2:

```mojo
from std.memory import Layout, Allocation, alloc, dealloc
from std.random import rand

struct Matrix[dtype: DType]:
    var rows: Int
    var cols: Int
    var _alloc: Allocation[Scalar[Self.dtype]]

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self._alloc = alloc(Layout[Scalar[Self.dtype]](count=rows * cols))

    def __deinit__(deinit self):
        dealloc(self._alloc^)

    def data(ref self) -> type_of(self._alloc.unsafe_ptr()):
        return self._alloc.unsafe_ptr()

    def __getitem__(self, row: Int, col: Int) -> Scalar[Self.dtype]:
        return self._alloc.unsafe_ptr()[unsafe_offset=row * self.cols + col]

    def __setitem__(mut self, row: Int, col: Int, value: Scalar[Self.dtype]):
        self._alloc.unsafe_ptr()[unsafe_offset=row * self.cols + col] = value

    def rand_init(mut self):
        rand(self._alloc.unsafe_ptr(), self.rows * self.cols)
```

This version differs from the 2025 one in several ways:

- **The matrix owns an `Allocation`** instead of a raw pointer. `alloc(Layout[T](count=n))` returns an `Allocation[T]`, which the compiler forces you to dispose of. Forget the `dealloc` and the code doesn't compile, so the old leak-or-double-free risk is gone.
- **`__del__(owned self)` became `__deinit__(deinit self)`.**
- **Struct parameters must be written `Self.dtype`** inside the struct body.
- **`Scalar[dtype]`** is the idiomatic spelling of `SIMD[dtype, 1]`.
- **`data()` returns a pointer tied to the matrix's origin**, so the compiler knows the pointer is only valid while the matrix is alive.
- **`rand_init()`** fills the matrix with uniform random values.

The matrix uses **row-major storage**: elements are stored row by row in contiguous memory. This layout choice becomes crucial when we optimize for cache performance later.

## The Naive Approach: Row Mean Calculation

Here's a straightforward row mean. The functions now take the `Matrix` directly and return a `List`, so there's no manual `free()` for the caller to forget:

```mojo
def row_mean_naive[dtype: DType](mat: Matrix[dtype]) -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]](length=mat.rows, fill=0)

    for row in range(mat.rows):
        var sum = Scalar[dtype](0)
        for col in range(mat.cols):
            sum += mat[row, col]
        result[row] = sum / Scalar[dtype](mat.cols)

    return result^
```

This processes one element at a time, which is exactly what SIMD is designed to accelerate. Each addition works on a single `Scalar[dtype]`, so the processor's vector units sit mostly idle. (The `^` at the end transfers ownership of the list to the caller instead of copying it.)

## SIMD Optimization: Vectorization and Parallelization

Now let's combine `vectorize` with `parallelize`:

```mojo
from std.algorithm import vectorize
from std.sys import simd_width_of
from max.algorithm import parallelize

def row_mean_optim[dtype: DType](mat: Matrix[dtype]) -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]](length=mat.rows, fill=0)
    var out = result.unsafe_ptr()
    var data = mat.data()
    var cols = mat.cols
    comptime simd_width = simd_width_of[dtype]()

    def row_mean_parallelized(row: Int) {imm out, imm data, imm cols}:
        var sum = Scalar[dtype](0)
        var row_ptr = data.unsafe_offset(row * cols)

        def row_mean_vectorized[width: Int](col: Int) {mut sum, imm row_ptr}:
            var simd_vals = row_ptr.unsafe_offset(col).unsafe_load[width=width]()
            sum += simd_vals.reduce_add()

        vectorize[simd_width](cols, row_mean_vectorized)
        out[unsafe_offset=row] = sum / Scalar[dtype](cols)

    parallelize(row_mean_parallelized, mat.rows)
    return result^
```

> **Heads-up:** `parallelize` no longer lives in the Mojo standard library. It moved to the MAX package: `from max.algorithm import parallelize`. Install `max` alongside `mojo` (for example `pixi add max` or `pip install modular`) or this import will fail.

### Breaking Down the Optimization

**1. SIMD Width Detection:** `simd_width_of[dtype]()` picks the best vector width for your CPU and data type at compile time. For `float32` it returns 8 on AVX2 (256-bit registers) and 4 on Apple Silicon's 128-bit NEON registers.

**2. Vectorization:** The inner `vectorize` call processes several columns at once:

```mojo
var simd_vals = row_ptr.unsafe_offset(col).unsafe_load[width=width]()
sum += simd_vals.reduce_add()
```

Instead of `width` separate load-and-add operations, we do one vector load and one horizontal reduction. Notice the capture list `{mut sum, ...}`: the closure needs to *modify* `sum`, and Mojo makes you say so explicitly.

**3. Parallelization:** The outer `parallelize` call spreads rows across CPU cores. That adds thread-level parallelism on top of data-level parallelism. The call is now `parallelize(closure, num_work_items)`, with the closure passed as an argument instead of a `[parameter]`.

## The Column Mean Challenge: Memory Layout Matters

Column operations are a different problem. Here's the naive approach:

```mojo
def col_mean_naive[dtype: DType](mat: Matrix[dtype]) -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]](length=mat.cols, fill=0)

    for col in range(mat.cols):
        var sum = Scalar[dtype](0)
        for row in range(mat.rows):
            sum += mat[row, col]  # Strided access!
        result[col] = sum / Scalar[dtype](mat.rows)

    return result^
```

The problem is the **memory access pattern**. Row operations read contiguous memory, which is cache-friendly. Column operations read elements that sit `cols` positions apart. This strided pattern causes cache misses and prevents straightforward SIMD loads.

### Attempt 1: Transpose First

The idea in the original version of this post was to transpose the matrix so that every column becomes a contiguous row, then reuse the row-mean approach:

```mojo
def col_mean_optim[dtype: DType](mat: Matrix[dtype]) -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]](length=mat.cols, fill=0)
    var out = result.unsafe_ptr()
    var data = mat.data()
    var rows = mat.rows
    var cols = mat.cols

    # Transpose into a scratch buffer so each column becomes contiguous
    var transposed = Matrix[dtype](cols, rows)
    var t_data = transposed.data()

    def transpose_parallelized(row: Int) {imm data, imm t_data, imm rows, imm cols}:
        for col in range(cols):
            t_data[unsafe_offset=col * rows + row] = data[unsafe_offset=row * cols + col]

    parallelize(transpose_parallelized, rows)

    comptime simd_width = simd_width_of[dtype]()

    def col_mean_parallelized(col: Int) {imm out, imm t_data, imm rows}:
        var sum = Scalar[dtype](0)
        var col_ptr = t_data.unsafe_offset(col * rows)

        def col_mean_vectorized[width: Int](row: Int) {mut sum, imm col_ptr}:
            var simd_vals = col_ptr.unsafe_offset(row).unsafe_load[width=width]()
            sum += simd_vals.reduce_add()

        vectorize[simd_width](rows, col_mean_vectorized)
        out[unsafe_offset=col] = sum / Scalar[dtype](rows)

    parallelize(col_mean_parallelized, cols)
    _ = transposed^  # keep the scratch buffer alive until the work is done
    return result^
```

Using a `Matrix` as the scratch buffer means it is freed automatically. The `_ = transposed^` line makes it explicit that the buffer has to outlive the parallel loops that read it through `t_data`.

It looks reasonable, but when I re-ran the benchmarks, **this version was slower than the naive one** (about 0.75x on an Apple M3 Pro). The transpose itself is a strided pass over the whole matrix: it writes with a stride of `rows`. So we pay the cache-miss cost we were trying to avoid, plus a 4 MB allocation, before doing any useful work.

### Attempt 2: Vectorize *Across* Columns

The better trick is to stop fighting the layout. We don't need each *column* to be contiguous. A *row* already holds many neighboring columns side by side. So we walk down the rows and add `simd_width` neighboring columns into a running column-sum vector in one instruction:

```mojo
from std.math import ceildiv

def col_mean_tiled[dtype: DType](mat: Matrix[dtype]) -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]](length=mat.cols, fill=0)
    var out = result.unsafe_ptr()
    var data = mat.data()
    var rows = mat.rows
    var cols = mat.cols
    comptime simd_width = simd_width_of[dtype]()
    comptime tile = 64  # columns handled by one task
    var num_tiles = ceildiv(cols, tile)

    def col_tile_parallelized(t: Int) {imm out, imm data, imm rows, imm cols}:
        var start = t * tile
        var tile_cols = min(tile, cols - start)
        var acc = out.unsafe_offset(start)

        # Walk down the rows; each row's slice of this tile is contiguous,
        # so we add `simd_width` neighbouring columns per instruction.
        for row in range(rows):
            var row_ptr = data.unsafe_offset(row * cols + start)

            def accumulate[width: Int](c: Int) {imm acc, imm row_ptr}:
                var partial = acc.unsafe_offset(c).unsafe_load[width=width]()
                var vals = row_ptr.unsafe_offset(c).unsafe_load[width=width]()
                acc.unsafe_offset(c).unsafe_store(partial + vals)

            vectorize[simd_width](tile_cols, accumulate)

        def finalize[width: Int](c: Int) {imm acc, imm rows}:
            var sums = acc.unsafe_offset(c).unsafe_load[width=width]()
            acc.unsafe_offset(c).unsafe_store(sums / Scalar[dtype](rows))

        vectorize[simd_width](tile_cols, finalize)

    parallelize(col_tile_parallelized, num_tiles)
    return result^
```

Each task owns a tile of 64 columns, so threads never write to the same output element. Within a tile, every load is contiguous, and there is no transpose and no extra allocation. There's also no horizontal `reduce_add`: each SIMD lane *is* one column's running sum. This version is about **6x faster** than the naive loop.

## Benchmarking: Measuring Real Performance

Proper benchmarking is crucial for optimization work, as the transpose surprise above shows. Here is the complete harness.

### The Complete Benchmarking Code

```mojo
from std.random import seed
from std.benchmark import run, keep

def print_sample[dtype: DType](data: List[Scalar[dtype]], name: String):
    print(name + ":")
    var n = min(len(data), 10)
    if len(data) > 10:
        print("First 10 elements:")
    for i in range(n):
        print("  ", data[i])
    print()


def main() raises:
    # Set a seed for reproducible results
    seed(42)

    comptime matrix_size = 1024  # 1024x1024 float32 matrix = 4 MB of data
    var mat = Matrix[DType.float32](matrix_size, matrix_size)

    print("Filling matrix with random values...")
    mat.rand_init()
    print("Matrix filled.\n")

    # First, verify the implementations agree
    print_sample(row_mean_naive(mat), "Row Mean Naive")
    print_sample(row_mean_optim(mat), "Row Mean Optimized")
    print_sample(col_mean_naive(mat), "Col Mean Naive")
    print_sample(col_mean_optim(mat), "Col Mean Optimized (transpose)")
    print_sample(col_mean_tiled(mat), "Col Mean Optimized (tiled)")

    # Benchmark closures capture `mat` by reference; `keep` stops the
    # compiler from deleting work whose result is never used.
    def bench_row_mean_naive() {imm mat}:
        keep(row_mean_naive(mat))

    def bench_row_mean_optim() {imm mat}:
        keep(row_mean_optim(mat))

    def bench_col_mean_naive() {imm mat}:
        keep(col_mean_naive(mat))

    def bench_col_mean_optim() {imm mat}:
        keep(col_mean_optim(mat))

    def bench_col_mean_tiled() {imm mat}:
        keep(col_mean_tiled(mat))

    print("Running benchmarks...")
    var report_row_naive = run(bench_row_mean_naive, max_runtime_secs=2)
    var report_row_optim = run(bench_row_mean_optim, max_runtime_secs=2)
    var report_col_naive = run(bench_col_mean_naive, max_runtime_secs=2)
    var report_col_optim = run(bench_col_mean_optim, max_runtime_secs=2)
    var report_col_tiled = run(bench_col_mean_tiled, max_runtime_secs=2)

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)

    print("\nRow Mean Operations:")
    print("-" * 30)
    print("Naive Implementation:")
    report_row_naive.print()
    print("\nOptimized Implementation:")
    report_row_optim.print()
    print("Row Mean Speedup:", report_row_naive.mean() / report_row_optim.mean(), "x")

    print("\nColumn Mean Operations:")
    print("-" * 30)
    print("Naive Implementation:")
    report_col_naive.print()
    print("\nTranspose Implementation:")
    report_col_optim.print()
    print("\nTiled Implementation:")
    report_col_tiled.print()
    print("Column Speedup (transpose):", report_col_naive.mean() / report_col_optim.mean(), "x")
    print("Column Speedup (tiled):    ", report_col_naive.mean() / report_col_tiled.mean(), "x")
```

### Key Benchmarking Principles

**1. Anti-optimization protection:** The compiler will happily delete a computation whose result is never used, and then you are benchmarking nothing. The old post worked around this with a hand-rolled trick: sum the results and print them behind an impossible `if sum > 1e10`. The standard library now has a purpose-built tool for this:

```mojo
keep(row_mean_naive(mat))  # the result is "used", so the work can't be removed
```

`keep` (and its sibling `black_box`, for hiding *inputs* from the optimizer) live in `std.benchmark`.

**2. Reproducible results:** Seed the random number generator so every run sees the same data:

```mojo
seed(42)
```

**3. Verify before benchmarking:** Print sample outputs and check that the implementations agree. A fast wrong answer is worthless. Small differences in the last digit are expected, because the SIMD versions add the numbers in a different order.

**4. Let the library do the statistics:** `run(f)` handles warm-up, repeated iterations, and statistics for you. It returns a `Report` with `.mean()` and `.print()`. One API change to note: the old `benchmark.run[func]()` took the function as a compile-time *parameter*. Now you pass the closure as an argument: `run(func)`. Because it is a real closure, it can capture `mat` directly.

### Sample Output

Here's a real run on an Apple M3 Pro (11 cores, `simd_width_of[DType.float32]() == 4`), Mojo 1.2.0:

```
Running benchmarks...

==================================================
BENCHMARK RESULTS
==================================================

Row Mean Operations:
------------------------------
Naive Implementation:
--------------------------------------------------------------------------------
Benchmark Report (s)
--------------------------------------------------------------------------------
Mean: 0.0006562101265822786
Total: 0.259203
Iters: 395
Warmup Total: 0.000759
Fastest Mean: 0.0006561776649746193
Slowest Mean: 0.0006562424242424242


Optimized Implementation:
--------------------------------------------------------------------------------
Benchmark Report (s)
--------------------------------------------------------------------------------
Mean: 0.00011614418604651163
Total: 0.574333
Iters: 4945
Warmup Total: 0.000126
Fastest Mean: 0.00010908656957928802
Slowest Mean: 0.00012319894864537

Row Mean Speedup: 5.64996104341796 x

Column Mean Operations:
------------------------------
Naive Implementation:
--------------------------------------------------------------------------------
Benchmark Report (s)
--------------------------------------------------------------------------------
Mean: 0.0008281755485893416
Total: 0.264188
Iters: 319
Warmup Total: 0.00107
Fastest Mean: 0.000822874213836478
Slowest Mean: 0.00083344375


Transpose Implementation:
--------------------------------------------------------------------------------
Benchmark Report (s)
--------------------------------------------------------------------------------
Mean: 0.0010986776556776558
Total: 0.299939
Iters: 273
Warmup Total: 0.001091
Fastest Mean: 0.0010454558823529412
Slowest Mean: 0.0011515109489051093


Tiled Implementation:
--------------------------------------------------------------------------------
Benchmark Report (s)
--------------------------------------------------------------------------------
Mean: 0.00014256550116550115
Total: 0.305803
Iters: 2145
Warmup Total: 5.6e-05
Fastest Mean: 0.0001419804104477612
Slowest Mean: 0.00014315004659832246

Column Speedup (transpose): 0.7537930204637951 x
Column Speedup (tiled):     5.809088045977763 x
```

### Understanding the Results

Across several runs on this machine:

| Operation | Naive | Optimized | Speedup |
|---|---|---|---|
| Row mean (vectorize + parallelize) | ~0.65 ms | ~0.10–0.13 ms | **5–7x** |
| Column mean, transpose first | ~0.82 ms | ~1.0–1.2 ms | **0.7–0.85x** (slower!) |
| Column mean, tiled across columns | ~0.82 ms | ~0.08–0.14 ms | **6–11x** |

- **Row operations** speed up nicely because they read contiguous memory.
- **The transpose trick doesn't pay off at this size.** The transpose is itself a strided pass, so it costs about as much as the naive column loop it was meant to replace.
- **Working with the layout beats working around it.** The tiled column mean reads memory in the same order as the row mean, and it is just as fast.
- **The run-to-run spread is real.** Parallel timings in the 0.1 ms range are sensitive to thread scheduling, so look at the range over several runs rather than a single number.

## Performance Expectations

For a 1024x1024 matrix of `float32` values:

- **Row operations:** roughly 5–7x from SIMD plus parallelization.
- **Column operations:** roughly 6–10x, *if* you vectorize across columns. A transpose-first approach may even lose to the naive loop.
- **Memory bandwidth** often becomes the limiting factor for larger matrices.

The exact numbers depend on your CPU's SIMD width (4 lanes on NEON, 8 on AVX2, 16 on AVX-512), its core count, and its memory system. Run the benchmark on your own machine.

## Key Takeaways

- **SIMD shines with regular, parallel computations** like matrix operations, image processing, and numerical algorithms.
- **Memory layout matters enormously.** Contiguous access enables efficient vectorization. Strided access needs a different algorithm, and the best one usually works *with* the layout.
- **Mojo's `vectorize` and `parallelize`** give clean abstractions for data-level and thread-level parallelism. In Mojo 1.2, closures are ordinary values with explicit capture lists.
- **Not every "optimization" is one.** The transpose looked clever and was slower. The fix came from rethinking the access pattern.
- **Always benchmark real workloads.** Theoretical speedups don't always appear in practice, because of memory bandwidth, cache effects, and thread overhead.

## Going Further

This example covers the fundamentals, but there's much more to explore:

- **Cache-blocked (tiled) transpose:** a blocked transpose can make Attempt 1 competitive. Try it and measure.
- **Vector accumulators:** keep a `SIMD[dtype, simd_width]` running sum in `row_mean_optim` and call `reduce_add()` once at the end, instead of once per chunk.
- **Loop tiling** for large matrices that don't fit in cache.
- **Mixed precision** to save memory bandwidth.

SIMD optimization is both an art and a science. Mojo gives you powerful abstractions and direct hardware access, so you can get remarkable performance while keeping the code clear. The key is to understand your data patterns, measure real performance, and let the hardware guide your decisions.

Start with these matrix operations, measure the speedups on your own system, and use the patterns as building blocks for more complex optimizations. The performance gains are worth the effort!

---

## Appendix: What Changed from the 2025 Version

If you wrote Mojo code from the original post, here's the migration checklist for Mojo 1.2.0:

| Before (2025) | Mojo 1.2.0 |
|---|---|
| `fn` | `def` (`fn` was removed) |
| `alias N = ...` | `comptime N = ...` |
| `@parameter fn f[w: Int](i: Int):` | `def f[w: Int](i: Int) {mut x, imm y}:` (explicit capture list) |
| `vectorize[f, width](size)` | `vectorize[width](size, f)` |
| `from algorithm import parallelize` | `from max.algorithm import parallelize` (requires the MAX package) |
| `parallelize[f](n, n)` | `parallelize(f, n)` |
| `simdwidthof[dtype]()` | `simd_width_of[dtype]()` from `std.sys` |
| `UnsafePointer[T]` | `ImmPointer[T, _]` / `MutPointer[T, _]` |
| `UnsafePointer[T].alloc(n)` / `.free()` | `alloc(Layout[T](count=n))` → `Allocation[T]`, released with `dealloc(a^)` |
| `ptr[i]`, `ptr + i` | `ptr[unsafe_offset=i]`, `ptr.unsafe_offset(i)` |
| `ptr.load[width=w](i)` / `ptr.store[width=w](i, v)` | `ptr.unsafe_offset(i).unsafe_load[width=w]()` / `.unsafe_store(v)` |
| `fn __del__(owned self)` | `def __deinit__(deinit self)` |
| `dtype` inside a struct | `Self.dtype` |
| `SIMD[dtype, 1]` | `Scalar[dtype]` (same type, clearer name) |
| `benchmark.run[f]()` | `run(f)` from `std.benchmark` |
| "impossible `if`" anti-optimization trick | `keep(result)` from `std.benchmark` |
| `from random import seed` | `from std.random import seed` (all stdlib imports use the `std.` prefix) |

The full, runnable source is a single file. Run it with `mojo run simd_matrix.mojo` in an environment that has both `mojo` and `max` installed.
