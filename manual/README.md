# Mojo Manual

My notes, exercises and progress while working through the Mojo Manual (Mojo **1.2.0**).

- Official manual: <https://mojolang.org/docs/manual/>

## Setup

```bash
pixi install            # creates the env from pixi.lock
pixi run version        # Mojo 1.2.0
pixi run mojo run p01_basics/s01_hello.mojo
pixi run mojo build p01_basics/s01_hello.mojo -o build/hello
```

## Layout

```
manual/
├── pixi.toml / pixi.lock   # one env for everything, pinned to mojo 1.2.*
├── pNN_<topic>/
│   ├── README.md           # notes, step checklist, checkpoint answers
│   ├── sNN_<topic>.mojo    # one file per guide step
│   ├── exercises/          # "Try it" tasks and checkpoint exercises
│   └── errors/             # code that should NOT compile, with the error noted
├── projects/               # capstones that mix several parts
└── scratch/                # playground (gitignored)
```

## Progress

| Part | Topic | Steps | Done | Checkpoint |
|---|---|:---:|:---:|:---:|
| 1 | [Language basics](p01_basics/README.md) | 14 | 0 | [ ] |
| 2 | [Language basics II: structs, traits, errors and packages](p02_structs_traits/README.md) | 14 | 0 | [ ] |
| 3 | [Value ownership](p03_ownership/README.md) | 14 | 0 | [ ] |
| 4 | [Value lifecycle](p04_lifecycle/README.md) | 14 | 0 | [ ] |
| 5 | [Metaprogramming](p05_metaprogramming/README.md) | 14 | 0 | [ ] |
| 6 | [Pointers](p06_pointers/README.md) | 14 | 0 | [ ] |
| 7 | [Advanced functions](p07_functions/README.md) | 14 | 0 | [ ] |
| 8 | [Python and C interop](p08_interop/README.md) | 14 | 0 | [ ] |
| | **Total** | **112** | **0** | |

Tick steps in each part's README, then update the Done column here.

## Log

| Date | Part / steps | Notes |
|---|---|---|
| | | |
