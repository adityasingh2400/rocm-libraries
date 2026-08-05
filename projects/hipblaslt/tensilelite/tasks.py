# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

from invoke.exceptions import Exit
from invoke.tasks import task
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

_TASKS_DIR = pathlib.Path(__file__).parent.resolve()


def _cmake_bool(value):
    return "ON" if value else "OFF"


def _detect_rocm():
    """Detect ROCm installation path.

    Priority: ROCM_PATH env > rocm-sdk path --root > /opt/rocm.
    """
    env_path = os.environ.get("ROCM_PATH")
    if env_path:
        return env_path

    if shutil.which("rocm-sdk"):
        try:
            result = subprocess.check_output(
                ["rocm-sdk", "path", "--root"], stderr=subprocess.DEVNULL
            ).decode().strip()
            if result:
                return result
        except subprocess.CalledProcessError:
            pass

    return "/opt/rocm"


def detect_gpu_arch():
    try:
        result = subprocess.run(["rocm_agent_enumerator", "-v"], capture_output=True, text=True, timeout=5, check=True)
        if result.returncode == 0:
            target = next((line.strip() for line in result.stdout.splitlines() if line.startswith("gfx") and line.strip() != "gfx000"), None)
            if target:
                return target
    except FileNotFoundError:
        print("Error: 'rocm_agent_enumerator' command not found. Please install ROCm.", file=sys.stderr)

    except subprocess.TimeoutExpired:
        print("Error: GPU detection timed out. Hardware might be unresponsive.", file=sys.stderr)

    except Exception as e:
        print(f"An unexpected error occurred during GPU detection: {e}", file=sys.stderr)

    print(f"Failed to detect a valid GPU architecture (gfx target not found).", file=sys.stderr)
    return None

@task
def get_gpu_arch(c):
    print(detect_gpu_arch())

@task(
    help={
        "rocisa_dir": "Path to the rocisa source directory (default: rocisa/ next to this file).",
        "stinkytofu_prefix": "Install prefix for the stinkytofu build (default: build_tmp/stinkytofu-install).",
    }
)
def rocisa(c, rocisa_dir=None, stinkytofu_prefix=None):
    """Install rocisa editably for source development.

    This is a separate rocisa developer workflow. TensileLite packaging and
    ``invoke build-client`` consume an already importable rocisa and do not call
    this task or make decisions about rocisa's release packaging.
    """
    _pip_install_rocisa(c, rocisa_dir, stinkytofu_prefix)


def _load_stinkytofu_tasks():
    """Import shared/stinkytofu/tasks.py without triggering its venv guard."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "stinkytofu_tasks",
        _TASKS_DIR.parent.parent.parent / "shared" / "stinkytofu" / "tasks.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_and_install_stinkytofu(c, install_prefix: pathlib.Path, rocm: str) -> None:
    """Build the rocisa development dependency into a private prefix."""
    stinkytofu_src = _TASKS_DIR.parent.parent.parent / "shared" / "stinkytofu"
    build_dir = install_prefix.parent / "stinkytofu-build"
    build_dir.mkdir(parents=True, exist_ok=True)

    rocm_s = str(rocm)
    cxx = shutil.which("amdclang++") or f"{rocm_s}/bin/amdclang++"
    cc = shutil.which("amdclang") or f"{rocm_s}/bin/amdclang"
    stinkytofu_tasks = _load_stinkytofu_tasks()
    cmake_cmd = [
        "cmake",
        "-S", str(stinkytofu_src),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DROCM_PATH={rocm_s}",
        f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DCMAKE_C_COMPILER={cc}",
        "-DSTINKYTOFU_INSTALL_RPATH_USE_LINK_PATH=ON",
        *stinkytofu_tasks.cmake_build_args(
            install_prefix=install_prefix, tests=False, python=False
        ),
    ]
    if shutil.which("ninja"):
        cmake_cmd.append("-G Ninja")
    if shutil.which("ccache"):
        cmake_cmd += [
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ]
    c.run(shlex.join(cmake_cmd))
    c.run(shlex.join(["cmake", "--build", str(build_dir), "--parallel"]))
    c.run(shlex.join(["cmake", "--install", str(build_dir)]))


def _pip_install_rocisa(c, rocisa_dir=None, stinkytofu_prefix=None):
    """Build and editable-install rocisa for its standalone developer flow."""
    src = pathlib.Path(rocisa_dir).resolve() if rocisa_dir else _TASKS_DIR / "rocisa"
    rocm = _detect_rocm()
    prefix = (
        pathlib.Path(stinkytofu_prefix).resolve()
        if stinkytofu_prefix
        else _TASKS_DIR / "build_tmp" / "stinkytofu-install"
    )
    _build_and_install_stinkytofu(c, prefix, rocm)

    cmake_args = f"-DROCM_PATH={rocm} -DROCISA_INCLUDE_BUILD_INFO=ON"
    if shutil.which("ccache"):
        cmake_args += (
            " -DCMAKE_C_COMPILER_LAUNCHER=ccache"
            " -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
        )
    env = dict(os.environ, CMAKE_ARGS=cmake_args)
    existing_prefix = env.get("CMAKE_PREFIX_PATH")
    env["CMAKE_PREFIX_PATH"] = (
        f"{prefix}{os.pathsep}{existing_prefix}" if existing_prefix else str(prefix)
    )
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 1))
    c.run(f"pip install --no-build-isolation -e {shlex.quote(str(src))}", env=env)


@task(
    help={
        "clean": "Remove the client build directory before building.",
        "configure": "Run CMake configuration for the client.",
        "build": "Build the tensilelite-client executable.",
        "build_dir": "Path to client build dir.",
        "build_type": "CMake build type (e.g. Release, Debug).",
        "gpu_targets": "Comma-separated list of GPU targets (e.g. gfx90a,gfx1101).",
        "rocm_path": "Path to a ROCm install whose amdclang/amdclang++ should be used.",
        "export_compile_commands": "Enable CMAKE_EXPORT_COMPILE_COMMANDS.",
        "enable_rocprof": "Build tensilelite-client with rocprof.",
        "cxx_flags_release": "Override CMAKE_CXX_FLAGS_RELEASE (for example, -O3 to keep asserts enabled in Release).",
        "enable_asan": "Enable AddressSanitizer.",
        "enable_tsan": "Enable ThreadSanitizer.",
    }
)
def build_client(
    c,
    clean=False,
    configure=True,
    build=True,
    build_dir="build_tmp",
    build_type="Release",
    gpu_targets=None,
    rocm_path=None,
    export_compile_commands=False,
    enable_rocprof=False,
    cxx_flags_release=None,
    enable_asan=False,
    enable_tsan=False,
):
    """Build the tensilelite-client C++ executable.

    The build stages the client and installs TensileLite into a private Python
    environment. rocisa must already be importable in the invoking Python
    environment and is inherited as an external dependency.
    """

    if enable_asan and enable_tsan:
        raise Exit("Error: ASAN and TSAN cannot be enabled simultaneously", code=1)

    if gpu_targets is None:
        gpu_targets = detect_gpu_arch()
        if not gpu_targets:
            raise Exit("Error: No GPU detected and no gpu_targets provided", code=1)
        print(f"warning: No GPU targets specified. Detected and using: {gpu_targets}")

    if rocm_path:
        cmake_c_compiler = os.path.join(rocm_path, "bin", "amdclang")
        cmake_cxx_compiler = os.path.join(rocm_path, "bin", "amdclang++")

        for compiler in (cmake_c_compiler, cmake_cxx_compiler):
            try:
                subprocess.run([compiler, "--version"], capture_output=True, timeout=5, check=True)
            except FileNotFoundError:
                raise Exit(f"Error: compiler not found at {compiler}", code=1)
            except subprocess.SubprocessError as e:
                raise Exit(f"Error: compiler check failed for {compiler}: {e}", code=1)

    if clean and os.path.exists(build_dir):
        c.run(f"rm -rf {shlex.quote(build_dir)}")

    if configure:
        os.makedirs(build_dir, exist_ok=True)

        cmake_cmd = [
            "cmake",
            "--preset",
            "tensilelite",
            "-S", str(_TASKS_DIR.parent),
            "-B", build_dir,
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DGPU_TARGETS={gpu_targets}",
            f"-DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK={_cmake_bool(enable_rocprof)}",
        ]

        if cxx_flags_release is not None:
            cmake_cmd.append(f"-DCMAKE_CXX_FLAGS_RELEASE={cxx_flags_release}")
        if rocm_path:
            cmake_cmd.append(f"-DCMAKE_C_COMPILER={cmake_c_compiler}")
            cmake_cmd.append(f"-DCMAKE_CXX_COMPILER={cmake_cxx_compiler}")
        if shutil.which("ccache"):
            cmake_cmd.append("-DCMAKE_C_COMPILER_LAUNCHER=ccache")
            cmake_cmd.append("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
        if export_compile_commands:
            cmake_cmd.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
        if enable_asan:
            cmake_cmd.append("-DTENSILELITE_ENABLE_HOST_ASAN=ON")
        if enable_tsan:
            cmake_cmd.append("-DTENSILELITE_ENABLE_HOST_TSAN=ON")
        cmake_cmd.append("-DHIPBLASLT_BUNDLE_PYTHON_DEPS=OFF")
        cmake_cmd.append("-DHIPBLASLT_TENSILELITE_PYTHON_MODE=BUILD")

        c.run(shlex.join(cmake_cmd))

    if build:
        c.run(shlex.join(["cmake", "--build", build_dir, "--parallel"]))


@task
def precommit_install(c):
    """Install the hipblaslt/TensileLite git pre-commit hook (run once after `uv sync`).

    Clears core.hooksPath only when it points at the default hooks dir; bails if
    it points somewhere custom.
    """
    root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], text=True, cwd=root
    ).strip()
    common_path = pathlib.Path(common)
    if not common_path.is_absolute():
        common_path = (pathlib.Path(root) / common_path).resolve()
    default_hooks = str(common_path / "hooks")
    hooks_path = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=root, capture_output=True, text=True,
    ).stdout.strip()

    if hooks_path:
        if os.path.realpath(hooks_path) == os.path.realpath(default_hooks):
            print(f"core.hooksPath is set to the default ({hooks_path}); clearing it "
                  "(redundant; git-lfs hooks are unaffected).")
            with c.cd(root):
                c.run("git config --unset-all core.hooksPath")
        else:
            raise Exit(
                f"Refusing to install: core.hooksPath is set to a custom path "
                f"({hooks_path}), not the default ({default_hooks}). Resolve that "
                "first.",
                code=1,
            )

    config = "projects/hipblaslt/.pre-commit-config.yaml"
    with c.cd(root):
        c.run(f"pre-commit install --config {shlex.quote(config)}", pty=True)


@task(
    help={
        "build_dir": "Path to coverage build dir.",
        "gpu_targets": "GPU targets (e.g. gfx90a,gfx942).",
        "rocm_path": "Path to ROCm installation.",
        "clean": "Remove build directory before building.",
    }
)
def build_coverage(
    c,
    build_dir="build_cov",
    gpu_targets=None,
    rocm_path=None,
    clean=False,
):
    """Build TensileLite with code coverage instrumentation.

    Builds rocisa, tensilelite-host, and client with LLVM coverage flags.
    Run tests with tox -e coverage-cpp to generate coverage reports.
    """
    if gpu_targets is None:
        gpu_targets = detect_gpu_arch()
        if not gpu_targets:
            print("Error: No GPU detected and no gpu_targets provided.")
            return

    if clean and os.path.exists(build_dir):
        c.run(f"rm -rf {shlex.quote(build_dir)}")

    os.makedirs(build_dir, exist_ok=True)

    rocm = rocm_path or _detect_rocm()
    cmake_c = os.path.join(rocm, "bin", "amdclang")
    cmake_cxx = os.path.join(rocm, "bin", "amdclang++")

    cmake_cmd = [
        "cmake",
        "--preset", "tensilelite",
        "-S", "../",
        "-B", build_dir,
        "-DCMAKE_BUILD_TYPE=Debug",
        f"-DGPU_TARGETS={gpu_targets}",
        f"-DCMAKE_C_COMPILER={cmake_c}",
        f"-DCMAKE_CXX_COMPILER={cmake_cxx}",
        "-DTENSILELITE_ENABLE_COVERAGE=ON",
        "-DROCISA_ENABLE_COVERAGE=ON",
        "-DHIPBLASLT_BUNDLE_PYTHON_DEPS=ON",
        "-DTENSILELITE_BUILD_TESTING=ON",
        "-DHIPBLASLT_ENABLE_YAML=OFF",  # Use msgpack, LLVM headers may not be available
    ]

    if shutil.which("ccache"):
        cmake_cmd.extend([
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ])

    c.run(shlex.join(cmake_cmd))
    c.run(shlex.join(["cmake", "--build", build_dir, "--parallel"]))
