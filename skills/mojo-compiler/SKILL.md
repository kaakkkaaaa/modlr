---
name: mojo-compiler
---

# Mojo compiler

Flags go before the source path. Tokens after the path are the program's arguments.

```bash
mojo run -O0 -g file.mojo --app-flag
mojo file.mojo          # same as mojo run
mojo                    # REPL
mojo --version
```

Fetch `https://mojolang.org/docs/cli/<command>.md` before inventing a flag. `mojo <command> --help` is the local source of truth.

Source: <https://mojolang.org/docs/cli/>. Full text: <https://mojolang.org/llms-cli.txt>.

## Commands

| Command | What it does |
|---|---|
| `run` | Compile and execute. `mojo file.mojo` omits `run`. |
| `build` | Write an executable, or another artifact with `--emit`. |
| `format` | Format sources. The command is `format`, not `fmt`. |
| `debug` | LLDB, or cuda-gdb with `--cuda-gdb`. |
| `doc` | Docstrings to JSON. Not HTML. |
| `precompile` | A directory with `__init__.mojo` becomes `.mojoc` or `.mojopkg`. |
| `demangle` | Demangle one symbol name. |
| `repl` | REPL. Also `mojo` with no arguments. |

These commands do not exist: `mojo test`, `mojo check`, `mojo fmt`, `mojo package`, `mojo bench`. Tests are ordinary programs. See `mojo-stdlib`.

## Common invocations

```bash
mojo build app.mojo                         # writes ./app
mojo build app.mojo -o bin/app
mojo build lib.mojo --emit shared-lib -o libfoo.so
mojo format -l 80 src/
mojo doc pkg/ -o api.json
mojo precompile mypackage -o mypack.mojoc
mojo --clear-cache -f
mojo --print-cache-location
```

`--emit` values:

| Value | Output | Cross-compile |
|---|---|---|
| `exe` | Executable. The default. | Fails at link |
| `shared-lib` | Dynamic library | Fails at link |
| `object` | Relocatable object | Works |
| `asm` | Assembly, plus GPU sidecars | Works |
| `llvm` | LLVM IR | Works |
| `llvm-bitcode` | LLVM bitcode | Works |

A shared library must not contain `main`. Python libraries are not linked into the binary. A `.mojoc` loads only with the compiler that built it.

## Flags

| Flag | Meaning |
|---|---|
| `-O`, `--optimization-level` | `0`–`3`. Default is `3`. `--no-optimization` is `-O0`. |
| `-g`, `--debug-level` | `none` (default), `line-tables`, or `full`. |
| `-I PATH` | Extra directory for Mojo imports. |
| `-D KEY=VALUE` | Compile-time define, read from `std.defines`. |
| `-j`, `--num-threads` | Compile parallelism. `0` uses all cores. |
| `--target-accelerator` | GPU arch, such as `sm_90` or `gfx942`. |
| `--print-supported-accelerators` | List GPU arch names and exit. |
| `--print-effective-target` | Show the resolved target and exit. |
| `--target-triple`, `--march`, `--mcpu` | Cross target. `--print-supported-cpus` needs `--target-triple`. |
| `--sanitize address` or `thread` | ASan or TSan. `--shared-libasan` links ASan dynamically. |
| `--fp-mode contract=fast` or `off` | `fast` (default) contracts into FMA. |
| `--diagnostic-format text` or `json` | Diagnostic format. |
| `-Werror`, `--disable-warnings` | Treat warnings as errors, or hide them. |
| `--warn-on-unstable-apis` | Warn on unstable stdlib APIs. |
| `--diagnose-missing-doc-strings` | Warn on missing docs. |
| `--ignore-deprecated NAME` | Hide one deprecation. |
| `-Xlinker ARG`, `--lld-path` | Linker control. |
| `--experimental-fixit` | Rewrites source. Use only when the user asks. |

## Debug

`mojo debug file.mojo` builds with `-O0` and full debug info, then launches LLDB.

- LLDB does not see GPU code.
- `--cuda-gdb` steps into NVIDIA kernels and is weaker on the CPU. It is NVIDIA-only.
- `--break-on-launch` stops on the first instruction of each kernel.
- `--vscode` attaches to a running VS Code with the Mojo extension.
- `--pid` and `--process-name` attach to a running process.

## A failed compile

1. Rerun the user's command. Keep flags before the file.
2. A removed form (`fn`, `let`, `alias`, `read`, `@parameter`, `from gpu import`, `LayoutTensor`) is fixed with `mojo-manual` or `mojo-gpu`. Puzzle code is Mojo 1.2.0 too.
3. An unknown method or import is looked up on the stdlib or `max.gpu` page. Do not invent an overload.
4. A local package needs `-I`. `PYTHONPATH` does not add Mojo imports.
5. A `.mojoc` from another compiler version fails to load. Rebuild it. `--ignore-incompatible-precompiled-file-errors` only confirms that diagnosis.
