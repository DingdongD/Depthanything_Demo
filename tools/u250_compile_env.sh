#!/usr/bin/env bash
# Shared, location-independent DS-Compiler environment for U250 kernel builds.

u250_require_compile_env() {
  : "${DS_TOOLCHAIN_ROOT:?set DS_TOOLCHAIN_ROOT to the DS toolchain directory}"
  : "${PYTHON_ROOT:?set PYTHON_ROOT to the ACMLIR build directory}"

  DS_COMPILER=${DS_COMPILER:-$DS_TOOLCHAIN_ROOT/python_bin/compile.py}
  DS_COMPILER_PYTHON=${DS_COMPILER_PYTHON:-python3}
  DS_ARCH_16=${DS_ARCH_16:-$DS_TOOLCHAIN_ROOT/arch_16_mono.yaml}
  DS_ARCH_256=${DS_ARCH_256:-$DS_TOOLCHAIN_ROOT/arch_256_mono.yaml}
  ACOMPILER_EXTENSION_DIR=${ACOMPILER_EXTENSION_DIR:-$PYTHON_ROOT/RelWithDebInfo/lib}

  local required
  for required in "$DS_COMPILER" "$DS_ARCH_16" "$DS_ARCH_256"; do
    if [[ ! -f "$required" ]]; then
      printf 'missing required DS build input: %s\n' "$required" >&2
      return 2
    fi
  done
  if ! command -v "$DS_COMPILER_PYTHON" >/dev/null 2>&1 \
      && [[ ! -x "$DS_COMPILER_PYTHON" ]]; then
    printf 'DS compiler Python is not executable: %s\n' "$DS_COMPILER_PYTHON" >&2
    return 2
  fi
  if [[ ! -d "$ACOMPILER_EXTENSION_DIR" ]]; then
    printf 'ACOMPILER_EXTENSION_DIR is missing: %s\n' "$ACOMPILER_EXTENSION_DIR" >&2
    return 2
  fi

  compiler=$DS_COMPILER
  python_bin=$DS_COMPILER_PYTHON
  arch=$DS_ARCH_16,$DS_ARCH_256
  export DS_COMPILER DS_COMPILER_PYTHON DS_ARCH_16 DS_ARCH_256
  export PYTHON_ROOT ACOMPILER_EXTENSION_DIR compiler python_bin arch
}
