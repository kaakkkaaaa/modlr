# Install Mojo 🔥

Source: https://advent.mojolang.org/install.html

This workbook uses `mdBook` and the Mojo nightly build. Nightly gives you access to the latest Mojo features used throughout the exercises, but it also means you’ll need to keep your installation up to date as Mojo changes.

This chapter gets your development environment ready. You’ll check that your system supports Mojo, install the nightly build, add editor support, and verify that everything works. It also covers updating or removing Mojo when you need to.

![Mojo prepares to install itself.](install-light.png)

![Mojo prepares to install itself.](install-dark.png)

## System requirements

Ensure your system supports Mojo.

### macOS

macOS requirements:

- macOS Sequoia (15) or later
- Apple silicon (M1 or later)
- 8 GiB minimum for Mojo development.
- Xcode or Xcode Command Line Tools 16 or later on macOS.

### Linux

Linux requirements:

- glibc 2.34 or later. For example, Ubuntu 22.04 LTS or later.
- x86-64-v3 (Haswell-class or newer; CPUs from approximately 2013 onward) or ARM64 Neoverse N1 or newer (for example, AWS Graviton2 and later)
- 8 GiB RAM minimum for Mojo development
- A C compiler (such as cc, gcc, or clang), used as a linker.

### Windows

Windows isn't officially supported. Try Mojo on Windows using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu 22.04 LTS.

## Install

Pick an installer. Each one creates a project directory and virtual environment with Mojo.

### Install with pixi (recommended)

[Pixi](https://pixi.sh) handles package management and virtual environments with lock files and cross-language support.

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | sh

# Create project with Modular channels (Release)
pixi init advent-of-code \
  -c https://conda.modular.com/max/ -c conda-forge \
  && cd advent-of-code

# OR

# Create project with Modular channels (Nightly)
pixi init advent-of-code \
  -c https://conda.modular.com/max-nightly/ -c conda-forge \
  && cd advent-of-code

# Install Mojo
pixi add mojo

# Enter the pixi environment
pixi shell
```

Configure pixi to automatically use nightly Mojo builds:

```bash
echo 'default-channels = ["https://conda.modular.com/max-nightly",' \
     '"conda-forge"]' >> ~/.pixi/config.toml
```

### Install with uv

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project and environment
uv init advent-of-code && cd advent-of-code
uv venv && source .venv/bin/activate

# Install Mojo
uv pip install mojo \
  --index https://whl.modular.com/nightly/simple/ \
  --prerelease allow
```

**After installing**:

Run `source ~/.bashrc` (or `source ~/.zshrc` on Mac). This will refresh your paths.

Verify with `mojo --version`.

## VS Code extension

Install the [Mojo extension](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo) for syntax highlighting, code completion, and debugging. Also available on [Open VSX](https://open-vsx.org/extension/modular-mojotools/vscode-mojo). Verify the publisher is Modular.

## Update Mojo

- **pixi**: `pixi update mojo`
- **uv**: `uv sync --upgrade-package mojo`

## Uninstall Mojo

- **pixi**: `pixi remove mojo && exit`
- **uv**: `uv pip uninstall mojo && uv sync && deactivate`
