# Puzzle 05: Broadcast add

Build a matrix by adding a row vector to a column vector.

Each output element is one value from `a` plus one value from `b`, but the three tensors no longer share a shape. `a` is 1×2, `b` is 2×1, and `output` is 2×2. Each input needs its own indexing rule, and that rule is the whole idea of **broadcasting**.

## At a glance

| | |
|---|---|
| **Goal** | `output[row, col] = a[0, col] + b[row, 0]` |
| **Data** | `a`: 1×2, `b`: 2×1, `output`: 2×2 (`SIZE = 2`) |
| **Launch** | `grid_dim = 1` block, `block_dim = (3, 3)` threads |
| **Threads launched** | 9 (4 do work, 5 idle) |
| **New ideas** | broadcasting, tensors with different layouts, read-only inputs (`mut=False`), a kernel that is generic over `TensorEngine` |
| **File** | `p05.mojo` |

## What broadcasting means

If you have used NumPy or PyTorch, you have seen this:

```python
a = np.array([[1, 2]])        # shape (1, 2)
b = np.array([[0], [10]])     # shape (2, 1)
a + b                         # shape (2, 2) → [[1, 2], [11, 12]]
```

Neither input has the output's shape. The rule is: **an axis of length 1 is stretched to match the other operand.** `a` has one row, so that row is reused for every output row. `b` has one column, so that column is reused for every output column.

```
              a (1×2)
            ┌─────┬─────┐
            │  1  │  2  │   ← reused for every row
            └─────┴─────┘
 b (2×1)                         output (2×2)
 ┌─────┐                       ┌─────┬─────┐
 │  0  │ → reused for every    │  1  │  2  │   0 + [1, 2]
 ├─────┤   column              ├─────┼─────┤
 │ 10  │                       │ 11  │ 12  │  10 + [1, 2]
 └─────┘                       └─────┴─────┘
```

NumPy does not copy `a` and `b` into full 2×2 arrays before adding. The kernel here doesn't either. It **reads** from the smaller tensors with an index that ignores the length-1 axis.

## The three layouts

All three tensors are row-major `TileTensor`s, each with its own layout:

```mojo
comptime out_layout = row_major[SIZE, SIZE]()   # 2×2
comptime a_layout   = row_major[1, SIZE]()      # 1×2
comptime b_layout   = row_major[SIZE, 1]()      # 2×1
```

| Tensor | Layout | Shape | Buffer size | Filled with |
|---|---|---|---:|---|
| `a` | `row_major[1, SIZE]` | 1×2 | 2 | `a[i] = i + 1` → `[1, 2]` |
| `b` | `row_major[SIZE, 1]` | 2×1 | 2 | `b[i] = i * 10` → `[0, 10]` |
| `output` | `row_major[SIZE, SIZE]` | 2×2 | 4 | zeros, then the sums |

`a` and `b` are both 2 floats in memory. Only the layout says whether those 2 floats form a row or a column.

## The kernel

```mojo
def broadcast_add[
    Engine: TensorEngine,
](
    output: TileTensor[mut=True,  dtype, OutLayout, MutAnyOrigin,   Engine=Engine],
    a:      TileTensor[mut=False, dtype, ALayout,   ImmutAnyOrigin, Engine=Engine],
    b:      TileTensor[mut=False, dtype, BLayout,   ImmutAnyOrigin, Engine=Engine],
    size_dev: Int32,
) where (Engine.element_size == 1):
    var size = Int(size_dev)
    var row = thread_idx.y
    var col = thread_idx.x
    # FILL ME IN (roughly 2 lines)
    if row < size and col < size:
        output[row, col] = a[0, col] + b[row, 0]
```

### The indexing

| Read | Why |
|---|---|
| `a[0, col]` | `a` has only row 0. Every output row uses it, so the row index is fixed at 0 and only `col` changes. |
| `b[row, 0]` | `b` has only column 0. Every output column uses it, so the column index is fixed at 0 and only `row` changes. |

Each thread computes one output cell from one cell of `a` and one cell of `b`:

| Thread `(row, col)` | `a[0, col]` | `b[row, 0]` | `output[row, col]` |
|---|---:|---:|---:|
| `(0, 0)` | 1 | 0 | 1 |
| `(0, 1)` | 2 | 0 | 2 |
| `(1, 0)` | 1 | 10 | 11 |
| `(1, 1)` | 2 | 10 | 12 |

Several threads read the same input element. `a[0, 0]` is read by both `(0, 0)` and `(1, 0)`. That is fine: concurrent **reads** of the same location are safe. Only concurrent writes to the same location would need coordination, and every thread writes a different output cell.

The guard is the same 2D guard as [puzzle 04](../p04/README.md). The block is 3×3, so 5 of the 9 threads fall outside the 2×2 output and return.

### Read-only inputs

This is the first kernel that marks its inputs read-only:

| Parameter | `mut` | Origin | Meaning |
|---|---|---|---|
| `output` | `True` | `MutAnyOrigin` | The kernel writes here. |
| `a`, `b` | `False` | `ImmutAnyOrigin` | The kernel only reads. Writing `a[0, 0] = ...` is a compile error. |

In puzzles 01 to 04 every pointer was mutable, so an accidental write into an input would compile. Here the type system catches it. It also tells the reader which arguments are inputs and which are outputs.

### Why the kernel is generic over `Engine`

A `TileTensor`'s full type includes an `Engine` parameter, which describes the storage behind the view. `TileTensor(out, out_layout)` builds a tensor with a particular engine type, and the kernel's parameter types have to match it exactly.

Instead of writing that engine type out, the kernel takes it as a compile-time parameter, `[Engine: TensorEngine]`, and uses `Engine=Engine` on all three tensors. The `where (Engine.element_size == 1)` clause requires that each index holds exactly one scalar, which is what `+` on single elements assumes.

At the launch site, `main` passes the concrete engine of the output tensor:

```mojo
ctx.enqueue_function[broadcast_add[out_tensor.Engine]](
    out_tensor, a_tensor, b_tensor, Int32(SIZE),
    grid_dim=BLOCKS_PER_GRID,
    block_dim=THREADS_PER_BLOCK,
)
```

`broadcast_add[out_tensor.Engine]` makes one concrete kernel from the generic one, and that is what gets compiled for the GPU. All three tensors were created the same way (`TileTensor(device_buffer, layout)`), so they share the engine.

`p04-tile-tensor.mojo` did not need this, because it spelled its tensor type without an `Engine` argument and used `mut=True` for both tensors. Puzzle 05 is where the more precise signature starts. [Puzzle 07](../p07/README.md) uses it again.

## What `main` does

| Step | What happens |
|---|---|
| 1 | Allocate `out` (4 floats) and wrap it as `out_tensor` with `out_layout`. Print its shape: `out shape: 2 x 2`. |
| 2 | Allocate `expected` on the host and wrap it as `expected_tensor` with the same layout, so the reference can be written with `[i, j]` too. |
| 3 | Allocate `a` and `b`, 2 floats each. |
| 4 | Map both. Write `a[i] = i + 1` and `b[i] = i * 10`. Compute `expected_tensor[i, j] = a_host[j] + b_host[i]`, the same broadcast rule done on the CPU. |
| 5 | Wrap `a` and `b` as `TileTensor`s with their own layouts. |
| 6 | Launch `broadcast_add[out_tensor.Engine]`, synchronize, then compare the flat `out` buffer with `expected`. |

In step 4, `a_host[j]` goes with the **column** and `b_host[i]` goes with the **row**, which matches `a[0, col] + b[row, 0]` in the kernel.

## Expected output

```
out shape: 2 x 2
out: HostBuffer([1.0, 2.0, 11.0, 12.0])
expected: HostBuffer([1.0, 2.0, 11.0, 12.0])
Puzzle 05 complete ✅
```

The flat output `[1, 2, 11, 12]` is the 2×2 matrix `[[1, 2], [11, 12]]` in row-major order.

## Common mistakes

- **`a[col, 0]` or `b[0, row]`.** The indices are swapped. `a` has one row, so its **first** index must be 0. Getting this backwards reads past the end of a 2-element buffer.
- **`a[row, col]`.** This treats `a` as if it were full size. For `row = 1` it reads outside `a`.
- **Expecting `a + b` to be commutative in index terms.** The values add the same way, but `a` broadcasts along rows and `b` along columns. Swapping the layouts changes the result.
- **Writing to an input.** With `mut=False`, the compiler rejects it. That is the type doing its job.

## Try it yourself

1. Change the fill to `b[i] = i * 100` and predict the output before you run it.
2. Try `output[row, col] = a[0, col] * b[row, 0]` (an outer product) and update the `expected` line to match.
3. Add `a[0, 0] = 0.0` inside the kernel. Compilation fails with `no matching method in call to '__setitem__'`, because a `mut=False` tensor has no way to write.
4. Set `SIZE = 3`. The 3×3 block now covers the output exactly. Every layout follows `SIZE`, so nothing else needs to change.

## Key takeaways

- Broadcasting reuses a length-1 axis across the output. In a kernel, that means **fixing that index at 0** when reading the smaller tensor.
- Each tensor can have its own layout, and the layout is part of the type.
- Many threads may read the same input element. That is safe. Only shared writes need care.
- `mut=False` / `ImmutAnyOrigin` mark read-only inputs, and the compiler enforces them.
- A kernel can be generic over `TensorEngine`. The launch site picks the concrete engine with `kernel[tensor.Engine]`.

## Run

This folder is its own Pixi project. It pins Mojo 1.2.0 and MAX 26.7.0 on the stable MAX channel.

```bash
cd problems/p05
pixi install
pixi run mojo p05.mojo
```

`pixi install` is once.

---

Previous: [Puzzle 04: 2D map](../p04/README.md) · Next: [Puzzle 06: Blocks](../p06/README.md)
