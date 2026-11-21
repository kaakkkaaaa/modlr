from std.memory import Layout, Allocation, alloc, dealloc
from std.algorithm import vectorize
from std.sys import simd_width_of
from std.math import ceildiv
from std.random import seed, rand
from std.benchmark import run, keep
from max.algorithm import parallelize


# ---------------------------------------------------------------------------
# Warm-up examples
# ---------------------------------------------------------------------------


def add_arrays_scalar(
    a: ImmPointer[Float32, _],
    b: ImmPointer[Float32, _],
    result: MutPointer[Float32, _],
    size: Int,
):
    for i in range(size):
        result[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]


def add_arrays_simd(
    a: ImmPointer[Float32, _],
    b: ImmPointer[Float32, _],
    result: MutPointer[Float32, _],
    size: Int,
):
    comptime simd_width = 8  # Process 8 float32s at once

    def vectorized_add[width: Int](i: Int) {imm a, imm b, imm result}:
        var vec_a = a.unsafe_offset(i).unsafe_load[width=width]()
        var vec_b = b.unsafe_offset(i).unsafe_load[width=width]()
        var vec_result = vec_a + vec_b  # Add all pairs at once!
        result.unsafe_offset(i).unsafe_store(vec_result)

    vectorize[simd_width](size, vectorized_add)


def scale_array_naive(data: MutPointer[Float32, _], size: Int, factor: Float32):
    for i in range(size):
        data[unsafe_offset=i] = data[unsafe_offset=i] * factor


def scale_array_simd(data: MutPointer[Float32, _], size: Int, factor: Float32):
    comptime simd_width = simd_width_of[DType.float32]()

    def vectorized_scale[width: Int](i: Int) {imm data, imm factor}:
        var vec_data = data.unsafe_offset(i).unsafe_load[width=width]()
        var vec_factor = SIMD[DType.float32, width](factor)  # Broadcast
        data.unsafe_offset(i).unsafe_store(vec_data * vec_factor)

    vectorize[simd_width](size, vectorized_scale)


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Row mean
# ---------------------------------------------------------------------------


def row_mean_naive[dtype: DType](mat: Matrix[dtype]) -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]](length=mat.rows, fill=0)

    for row in range(mat.rows):
        var sum = Scalar[dtype](0)
        for col in range(mat.cols):
            sum += mat[row, col]
        result[row] = sum / Scalar[dtype](mat.cols)

    return result^


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


# ---------------------------------------------------------------------------
# Column mean
# ---------------------------------------------------------------------------


def col_mean_naive[dtype: DType](mat: Matrix[dtype]) -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]](length=mat.cols, fill=0)

    for col in range(mat.cols):
        var sum = Scalar[dtype](0)
        for row in range(mat.rows):
            sum += mat[row, col]  # Strided access!
        result[col] = sum / Scalar[dtype](mat.rows)

    return result^


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


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


def print_sample[dtype: DType](data: List[Scalar[dtype]], name: String):
    print(name + ":")
    var n = min(len(data), 10)
    if len(data) > 10:
        print("First 10 elements:")
    for i in range(n):
        print("  ", data[i])
    print()


def main() raises:
    # Warm-up examples
    var a: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    var b: List[Float32] = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    var c = List[Float32](length=8, fill=0)
    add_arrays_simd(a.unsafe_ptr(), b.unsafe_ptr(), c.unsafe_ptr(), len(a))
    print("A + B =", c)
    scale_array_simd(c.unsafe_ptr(), len(c), 2.0)
    print("2 * (A + B) =", c)
    print()

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
