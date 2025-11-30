# Mojo🔥 GPU Puzzles 🧩 - Solutions + Bonus 📦

My personal solutions to [Mojo GPU Puzzles](https://puzzles.modular.com/), a hands-on guide to GPU programming using Mojo🔥.

## About Mojo GPU Puzzles

Mojo GPU Puzzles is a practical, puzzle-based learning resource for mastering GPU programming with Mojo. The approach emphasizes learning through doing rather than extensive theory, with each puzzle building progressively on established concepts.

### Why This Matters

GPU programming has evolved from specialized skill to fundamental infrastructure for modern computing. From large language models to real-time computer vision, GPU acceleration drives computational breakthroughs across AI, scientific computing, autonomous systems, and financial analysis.

### Learning Through Puzzles

The puzzle-based methodology offers several advantages:
- **Direct experience**: Immediate execution on GPU hardware with concrete feedback
- **Incremental complexity**: Each challenge builds on previous concepts
- **Applied focus**: Problems mirror real-world computational scenarios
- **Diagnostic skills**: Systematic debugging practice
- **Knowledge retention**: Active problem-solving over passive consumption

## Repository Structure

```
.
├── problems/     # My puzzle implementations
│   ├── p01/     # Puzzle 1 solution
│   ├── p02/     # Puzzle 2 solution
│   └── ...
└── solutions/   # Reference solutions (optional)
```

## Progress Tracker

### Part I: GPU Fundamentals (Puzzles 1-8)
- [ ] Puzzle 1 - Map
- [ ] Puzzle 2 - Zip
- [ ] Puzzle 3 - Guard
- [ ] Puzzle 4 - Map 2D
- [ ] Puzzle 5 - Broadcast
- [ ] Puzzle 6 - Blocks
- [ ] Puzzle 7 - Shared Memory
- [ ] Puzzle 8 - Stencil

**Learning objectives**: Thread indexing, block organization, memory access patterns, LayoutTensor abstractions, shared memory basics

### Part II: Debugging GPU Programs (Puzzles 9-10)
- [ ] Puzzle 9 - GPU Debugger
- [ ] Puzzle 10 - Sanitizer

**Learning objectives**: GPU debugging techniques, sanitizers for memory errors and race conditions, systematic bug identification

*Note: Requires `pixi` and NVIDIA GPU with CUDA support*

### Part III: GPU Algorithms (Puzzles 11-16)
- [ ] Puzzle 11 - Reduction
- [ ] Puzzle 12 - Scan
- [ ] Puzzle 13 - Pool
- [ ] Puzzle 14 - Conv
- [ ] Puzzle 15 - Matmul
- [ ] Puzzle 16 - Flashdot

**Learning objectives**: Parallel reductions, pooling operations, convolution kernels, prefix sum algorithms, matrix multiplication with tiling

### Part IV: MAX Graph Integration (Puzzles 17-19)
- [ ] Puzzle 17 - Custom Op
- [ ] Puzzle 18 - Softmax
- [ ] Puzzle 19 - Attention

**Learning objectives**: Custom MAX Graph operations, GPU kernel integration with Python, production-ready operations

### Part V: PyTorch Integration (Puzzles 20-22)
- [ ] Puzzle 20 - Torch Bridge
- [ ] Puzzle 21 - Autograd
- [ ] Puzzle 22 - Fusion

**Learning objectives**: Bridging Mojo GPU kernels with PyTorch tensors, CustomOpLibrary usage, torch.compile integration, kernel fusion

### Part VI: Functional Patterns & Benchmarking (Puzzle 23)
- [ ] Puzzle 23 - Functional

**Learning objectives**: Functional patterns (elementwise, tiled processing, vectorization), systematic performance optimization, benchmarking skills

### Part VII: Warp-Level Programming (Puzzles 24-26)
- [ ] Puzzle 24 - Warp Sum
- [ ] Puzzle 25 - Warp Communication
- [ ] Puzzle 26 - Advanced Warp

**Learning objectives**: Warp fundamentals and SIMT execution, warp operations (sum, shuffle, broadcast), advanced shuffle patterns

### Part VIII: Block-Level Programming (Puzzle 27)
- [ ] Puzzle 27 - Block Operations

**Learning objectives**: Block-wide reductions, block-level prefix sum, intra-block coordination

### Part IX: Advanced Memory Systems (Puzzles 28-29)
- [ ] Puzzle 28 - Async Memory
- [ ] Puzzle 29 - Barriers

**Learning objectives**: Memory coalescing patterns, async memory operations, memory fences, synchronization primitives, prefetching strategies

### Part X: Performance Analysis & Optimization (Puzzles 30-32)
- [ ] Puzzle 30 - Profiling
- [ ] Puzzle 31 - Occupancy
- [ ] Puzzle 32 - Bank Conflicts

**Learning objectives**: GPU kernel profiling, bottleneck identification, occupancy optimization, shared memory bank conflict elimination

*Note: Requires NVIDIA GPU with NSight profiling tools*

### Part XI: Advanced GPU Features (Puzzles 33-34)
- [ ] Puzzle 33 - Tensor Cores
- [ ] Puzzle 34 - Cluster

**Learning objectives**: Tensor core programming for AI workloads, cluster programming in modern GPUs

*Note: Requires NVIDIA GPU with Tensor Core support*

## Setup Instructions

### Prerequisites

**System Requirements**:
- Compatible GPU ([check requirements](https://docs.modular.com/max/faq#gpu-requirements))
- For macOS Apple Silicon: macOS 15.0+, Xcode 16+ (currently supports puzzles 1-8, 11-15)

**Knowledge Requirements**:
- Programming fundamentals (variables, loops, conditionals, functions)
- Parallel computing concepts (threads, synchronization, race conditions)
- Basic Mojo familiarity ([Mojo manual](https://docs.modular.com/mojo/manual/))

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/modular/mojo-gpu-puzzles
cd mojo-gpu-puzzles
```

2. **Install package manager**:

**Option 1 (Recommended): pixi**
```bash
# Install
curl -fsSL https://pixi.sh/install.sh | sh

# Update
pixi self-update
```

**Option 2: uv**
```bash
# Install
curl -fsSL https://astral.sh/uv/install.sh | sh

# Update
uv self update

# Create virtual environment
uv venv && source .venv/bin/activate
```

*Note: Some puzzles require `pixi` for full functionality*

3. **Verify setup**:
```bash
# Check GPU specifications
pixi run gpu-specs

# Run first puzzle
pixi run p01
```

For AMD GPUs: `pixi run p01 -e amd`  
For Apple GPUs: `pixi run p01 -e apple`

## Working with Puzzles

### Workflow

1. Navigate to `problems/pXX/` to find the puzzle template
2. Implement your solution in the provided framework
3. Test your implementation: `pixi run pXX`
4. Compare with reference solutions to learn different approaches

### Essential Commands

```bash
# Run puzzles (NVIDIA by default)
pixi run pXX              # NVIDIA GPU
pixi run pXX -e amd       # AMD GPU
pixi run pXX -e apple     # Apple GPU

# Test solutions
pixi run tests            # Test all solutions
pixi run tests pXX        # Test specific puzzle

# Run manually
pixi run mojo problems/pXX/pXX.mojo   # Your implementation
pixi run mojo solutions/pXX/pXX.mojo  # Reference solution

# Interactive shell
pixi shell                # Enter environment
mojo problems/p01/p01.mojo
exit

# Development
pixi run format           # Format code
pixi task list            # Available commands
```

## GPU Platform Support

| Platform | Supported Puzzles | Notes |
|----------|------------------|-------|
| **NVIDIA GPU** | All (1-34) | Complete support, best learning experience |
| **AMD GPU** | 1-8, 11-29 | Missing debugging/profiling tools, Tensor Cores |
| **Apple GPU** | 1-8, 11-15 | Basic GPU programming patterns only |

### Platform-Specific Features

**NVIDIA GPUs**: Full curriculum access including debugging tools (9-10), profiling (30-32), and modern GPU features (33-34)

**AMD GPUs**: Comprehensive learning for algorithms and memory patterns, missing only vendor-specific tools

**Apple GPUs**: Suitable for fundamental GPU programming concepts

## Resources

- [Official Mojo GPU Puzzles](https://puzzles.modular.com/)
- [Mojo Documentation](https://docs.modular.com/mojo/manual/)
- [GPU Programming Fundamentals](https://docs.modular.com/mojo/manual/gpu/fundamentals)
- [Original GPU Puzzles (Python)](https://github.com/srush/GPU-Puzzles) - Inspiration for this project

## Acknowledgments

Parts I and III are heavily inspired by [GPU Puzzles](https://github.com/srush/GPU-Puzzles), an interactive NVIDIA GPU learning project. This adaptation reimplements concepts using Mojo's abstractions and expands on advanced topics with Mojo-specific optimizations.
