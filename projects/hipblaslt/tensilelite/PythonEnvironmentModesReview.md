<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Python Environment Modes Review

## Review status

**Changes requested — risk 4/5.**

The producer/consumer distinction behind two explicit Python environment modes
is sound, but the implementation introduced by commit `6382dcf61c` is not yet
defensible across its claimed build surface. The review found confirmed target
graph and preset failures, an incomplete Windows staging contract, several
important reproducibility and configuration gaps, and no automated replacement
coverage for a broad default-path build change.

This review was performed by independent staff-level Python packaging/runtime
and C++/CMake reviewers, followed by an adversarial integration-design pass.
No source files were changed as part of the review.

## Scope

Reviewed commit:

```text
547be905bc6c74afb5bf5240bfcf958feb24677d..6382dcf61cb65c1cbbef54b446ca618364997b1b
build(hipblaslt): stage the TensileLite Python environment
```

The commit changes:

- `../CMakeLists.txt`
- `../CMakePresets.json`
- `../cmake/HipBLASLtCodegen.cmake`
- `../cmake/hipblaslt_python.cmake`
- `tasks.py`
- deletion of `tensilelite/RocisaStatus.py`
- deletion of `tensilelite/Tests/unit/test_rocisa_install_status.py`

The current packaging decision documents were added after the implementation
commit. They were therefore treated as retrospective design intent rather than
proof that the original implementation met its acceptance criteria:

- [`docs/Public.md`](docs/Public.md)
- [`docs/PackagingDecisions.md`](docs/PackagingDecisions.md)
- [`docs/PackagingPlan.md`](docs/PackagingPlan.md)

Later decisions such as D17 and D18 are useful constraints for follow-up work,
but cannot fairly be used as original requirements for commit `6382dcf61c`.

## Intended domain boundaries

The design involves four distinct owners:

1. **Source producer:** the checkout supplies the pure-Python `tensilelite`
   package, generator resources, and release metadata.
2. **Installed consumer:** a selected Python environment and ROCm installation
   supply an installed, version-matched artifact set.
3. **rocisa owner:** rocisa has an independent native build, ABI, and release
   workflow. hipBLASLt should import and diagnose it, not implicitly decide how
   to build or package it.
4. **Native artifact owner:** CMake and ROCm packaging own
   `tensilelite-client`, compiler tools, headers, and runtime libraries.

The explicit two-workflow model is reasonable because source production and
installed consumption are genuinely different operations. Automatic detection
would make identical CMake commands behave differently depending on machine
state, while separate booleans would permit contradictory combinations.

The current names are less precise than the boundaries they represent:

- `BUILD` means a staged source environment, not simply “build.”
- `SYSTEM` can use any explicitly selected virtual environment, not necessarily
  the operating system Python.

If the interface is not yet externally committed, `STAGED|INSTALLED` would be
clearer. Otherwise, retain `BUILD|SYSTEM` as compatibility aliases and document
their precise semantics.

## Standards review

### Blocking: test-data generation can run before the private Python exists

The commit changes `HIPBLASLT_PYTHON_COMMAND` to point at:

```text
<build>/tensilelite-venv/bin/python
```

However, [`../clients/tests/CMakeLists.txt`](../clients/tests/CMakeLists.txt)
invokes that command to generate `hipblaslt_gtest.data` without depending on
`HIPBLASLT_PYTHON_DEPS`.

Consequences:

- A clean direct build of `hipblaslt-test` can invoke a nonexistent interpreter.
- A parallel default build has a scheduling race.
- The graph violates the decision that the environment must precede every
  Python generation command.

Adding `${HIPBLASLT_PYTHON_DEPS}` to the custom command would repair the direct
graph defect, but it would preserve an unnecessary module coupling.
`hipblaslt_gentest.py` is a generic hipBLASLt build script, not a TensileLite
package command. The preferable design is:

- use `Python3_EXECUTABLE` for generic hipBLASLt scripts;
- reserve a namespaced `HIPBLASLT_TENSILELITE_PYTHON_COMMAND` and dependency
  target for `python -m tensilelite ...` commands.

Transporting the command and dependency as separate global variables is also a
data-clump design smell. A CMake helper that creates TensileLite Python commands
with their dependency attached would make this class of error harder to repeat.

### Blocking: Windows `BUILD` mode creates an unusable ROCm root

On Windows, [`../cmake/hipblaslt_python.cmake`](../cmake/hipblaslt_python.cmake)
copies only `amdclang.exe` and `amdclang++.exe` into the synthetic ROCm root,
then sets `ROCM_PATH` to that root for every Python command.

TensileLite's Windows compiler implementation in
[`tensilelite/Toolchain/Component.py`](tensilelite/Toolchain/Component.py)
passes that value through `--rocm-path`. Clang expects HIP headers and runtime
libraries below that root, but they are not present in the stage.

This is a confirmed contract mismatch, although a Windows device-library build
was not available to execute during the review.

Preferred correction:

- keep `ROCM_PATH` pointing at the complete real SDK;
- bind the staged source-built client separately;
- do not represent a partial client stage as a complete ROCm SDK.

The less attractive alternative is to stage or junction every required Windows
SDK directory and maintain that layout as a supported contract.

### Important: `--rocm-path` does not select one coherent ROCm installation

`invoke build-client --rocm-path X` selects compilers below `X`, but does not
pass `-DROCM_PATH=X` to CMake. CMake can therefore compile with one SDK while
reading version metadata, staging toolchain links, and validating artifacts from
another SDK selected through the environment or `/opt/rocm` fallback.

The Invoke option must propagate a single coherent selection through:

- `ROCM_PATH`;
- `CMAKE_PREFIX_PATH`;
- C and C++ compiler paths;
- package version derivation;
- staged toolchain and runtime validation.

The build should diagnose disagreement between those inputs instead of silently
constructing a mixed-release artifact set.

### Important: `BUILD` is layered, not dependency-isolated

The private venv writes every parent `site-packages` or `dist-packages`
directory into `hipblaslt-bootstrap.pth`, installs TensileLite with
`--no-deps --no-build-isolation`, and imports a hand-maintained list of
dependencies.

This provides two useful properties:

- installing the current TensileLite checkout does not mutate the caller's
  Python environment;
- CMake performs no implicit network access.

It does **not** provide a hermetic dependency environment. All packages visible
in the selected parent environment can influence the build, mask undeclared
dependencies, or introduce conflicting versions.

The developer workflow should be described as a **build-private,
dependency-inheriting environment**. Release and CI validation should use a
separate clean environment populated from an offline, version-controlled
wheelhouse or lock.

If broad inheritance remains, tests must verify import origins, minimal declared
dependencies, conflicting versions, old build backends, and `pip check` results.

### Important: missing prerequisites fail too late

`SYSTEM` validates `import rocisa, tensilelite` during configuration. `BUILD`
does not discover a missing rocisa or build backend until the environment target
runs, potentially after compiling `tensilelite-client`.

`BUILD` should perform an early, opaque `import rocisa` check using the selected
Python. It should not interpret rocisa's version or native layout, because those
remain rocisa's responsibility. It should also verify the offline build/runtime
prerequisites needed by `--no-build-isolation --no-deps`, then retain the final
staged import as an artifact-contract check.

### Important: contributor documentation contradicts the new behavior

[`../CONTRIBUTING.md`](../CONTRIBUTING.md) still describes Python 3.8+, direct
use of the selected venv, the legacy `Tensile.tensilelite_create_library`
entrypoint, and no separately prepared rocisa requirement.

The implementation instead requires Python 3.10 for device generation, creates
a layered private venv, uses package commands, and expects rocisa to be supplied
independently. Contributor and migration documentation must change with the
implementation rather than in a later cleanup.

### Defensible cleanup: removal of rocisa auto-detection

Deleting `RocisaStatus.py`, the automatic rocisa rebuild/bundling decision, and
their tests is consistent with the decision that rocisa is independently
supplied. Those tests covered behavior that intentionally no longer exists.

The problem is not their deletion; the problem is the absence of replacement
tests for the new environment state machine and default build path.

## Spec and build-system review

### Blocking: the `hipblaslt-clients` preset is internally invalid

The `hipblaslt-clients` preset in
[`../CMakePresets.json`](../CMakePresets.json) enables device generation while
explicitly disabling `TENSILELITE_ENABLE_CLIENT`.

Default `BUILD` mode enters TensileLite environment setup because device
generation is enabled, then fatally requires a `tensilelite-client` target. The
preset therefore cannot configure as committed.

The immediate repair depends on the client-ownership decision below:

- if every generator invocation must require a client, enable or supply the
  client in every device-generation preset;
- if pure code generation is separated from benchmark/runtime validation,
  remove the unnecessary client prerequisite from generator setup.

### Blocking: no automated evidence covers the new default path

The commit changes all of the following without adding replacement automated
coverage:

- default build behavior;
- Python and rocisa ownership;
- target graph ordering;
- two Python environment states;
- device code-generation commands;
- shipped presets;
- Linux and Windows staging behavior;
- Python version and development-header requirements.

The later implementation plan records useful local results, but does not provide
exact commands, logs, shared-CI links, or an executable regression suite. It
also does not cover the target race or the required platform/configuration
matrix.

Under the ROCm PR quality rules, this broad default-path build change needs
shared CI evidence or an explicit, reviewer-approved waiver. No such evidence
was available during this review.

### Important: mandatory client validation is the wrong boundary for pure codegen

`logic` and `create-library` do not intrinsically execute
`tensilelite-client`, but importing TensileLite requires the client to be
present. Every source device-library build therefore compiles and stages the
benchmark client and its dependencies.

This coupling:

- broke a shipped preset;
- increases device-generation prerequisites and duration;
- conflates generator validity with release-bundle completeness;
- makes a Python generator import depend on an unrelated executable.

The statement that the artifacts form one release unit justifies validating the
complete release in packaging CI. It does not, by itself, justify requiring the
client for every logic-validation or library-generation process.

Recommended boundary:

- validate rocisa, ROCm compatibility, and generator resources for `logic` and
  `create-library`;
- validate `tensilelite-client` when `run` needs it;
- provide an explicit full-install validation command or packaging test for the
  version-matched release artifact set.

If import-time client validation is retained as a hard product invariant, the
team must explicitly accept the added C++ build dependency, update all presets,
and provide CI evidence that the coupling works across supported workflows.

### Important: multi-config builds share mutable staged state

One `_runtime_stamp`, venv, staged client, and artifact root are shared across
all configurations. Under Visual Studio or Ninja Multi-Config, Debug can
overwrite the staged client and a later Release build can incorrectly retain or
reuse it.

Either:

- make the stage, venv, client, and stamp configuration-specific; or
- explicitly reject multi-config `BUILD` until it is supported.

The first option is preferable if Windows multi-config builds are part of the
supported surface.

### Important: customized libexec paths break the fixed runtime contract

CMake stages the client below `${CMAKE_INSTALL_LIBEXECDIR}`, while Python's
runtime contract expects the fixed logical location:

```text
<ROCM_PATH>/libexec/hipblaslt/tensilelite/tensilelite-client
```

A nondefault or absolute `CMAKE_INSTALL_LIBEXECDIR` can therefore produce a
stage that the Python package cannot validate. The synthetic runtime layout
must use one fixed logical path, independent of customizable install
destinations, or both sides must consume one shared definition.

### Important: Python development headers are required unnecessarily

Ordinary `BUILD` and `SYSTEM` code generation consume an already-built rocisa
and a pure-Python TensileLite package. They require a Python interpreter, not
`Development.Module`.

Only the legacy in-tree `_rocisa` extension build needs Python development
headers. Requesting development headers for all device/client configurations
unnecessarily rejects otherwise valid minimal packaging environments and
weakens the independently supplied rocisa boundary.

### Important: coverage wiring contradicts the selected dependency model

`build_coverage` enables the bundled, instrumented rocisa target. The `BUILD`
venv, however, inherits rocisa from the selected external Python environment and
does not depend on the locally built `_rocisa` target.

The build can therefore compile an instrumented extension that code generation
and Python tests do not import. Without an external rocisa, the supposedly
bundled coverage build still fails.

Coverage should explicitly build/install/select the instrumented rocisa before
configuring the consuming environment, rather than combining mutually
independent dependency models.

### Additional important configuration gaps

- Staged venv, client, links, and metadata are not fully declared as CMake
  `BYPRODUCTS`.
- `/opt/rocm` is not a suitable universal fallback, especially on Windows.
- SYSTEM mode still tracks a source-tree known-bugs file even though the
  installed package resource is the file actually consumed.
- `TENSILELITE_ENABLE_CLIENT` defaults from device enablement even in SYSTEM,
  where the installed client is consumed and the locally built client may be
  unused.
- BUILD and SYSTEM mode validation is duplicated between the top-level CMake
  file and the helper function.

The source-tree known-bugs dependency is particularly misleading in SYSTEM:
editing the checkout can invalidate the stamp, but the rerun validates the
installed wheel's resource rather than the edited source resource. Dependencies
should be mode-specific.

## Architecture decision matrix

| Approach | Reproducibility | Developer UX | Packaging fidelity | Decision |
| --- | --- | --- | --- | --- |
| Keep the current implementation | Low to medium | Medium | Medium | Reject |
| Harden explicit staged/installed modes | High with strict CI; good locally | High | High | **Recommend** |
| Require only an external prepared environment | High | Low for raw source builds | High | Credible, but costly migration |
| Use a CMake `ExternalProject` venv | Medium to high | Medium | Medium to high | More complexity without fixing ownership |
| Select behavior only through presets | Low | Medium | Low for downstreams | Reject |
| Automatically detect source vs installed state | Low | Superficially high | Low | Reject |
| Strict build/install/test separation as the only workflow | Highest | Low for source iteration | Highest | Require in packaging CI, not as the sole developer path |

## Recommended decisions

### Keep two explicit workflows

Retain an explicit staged-source workflow and an explicit installed-artifact
workflow. This makes provenance observable and prevents machine-dependent
auto-detection.

### Keep rocisa externally owned

Do not restore automatic rocisa building or packaging decisions in the
TensileLite integration. Provide early diagnostics and documented preparation,
while leaving its ABI, native dependencies, and release format to the rocisa
workflow.

### Separate generator validity from complete-release validation

Do not make pure logic validation and device-library generation depend on the
benchmark client unless a concrete functional requirement is demonstrated.
Validate full artifact-set completeness explicitly in packaging CI and a
dedicated validation interface.

### Separate the Python command roles

Use the configured Python interpreter for generic build scripts. Use the
TensileLite package environment only for package commands and attach its target
dependency automatically through a helper abstraction.

### Make the ROCm root explicit and coherent

One selected root should own compilers, headers, libraries, release metadata,
and package compatibility. Staged client location is a separate concern and
must not turn a partial directory into the compiler SDK root.

### Support two dependency policies, not two ambiguous claims

- Developer STAGED mode may inherit prepared dependencies for fast offline
  iteration, but must say so explicitly.
- CI/release validation must install from a controlled no-index wheelhouse or
  equivalent lock into a clean environment.

Neither path should perform implicit network access.

## Required changes before merge

1. Separate ordinary build Python from the TensileLite package Python.
2. Fix or remove the broken `hipblaslt-clients` preset.
3. Separate the complete compiler SDK root from staged client/runtime artifacts,
   especially on Windows.
4. Propagate one explicit ROCm root through Invoke, CMake, compilers, versioning,
   and runtime validation.
5. Require `Development.Module` only for bundled rocisa builds.
6. Add early checks for rocisa and offline build prerequisites.
7. Document the dependency-inheritance policy and add a clean wheelhouse CI
   path.
8. Make staging configuration-aware and declare all byproducts.
9. Make resource and client dependencies mode-specific.
10. Update contributor and migration documentation.

## Minimum validation gate

Use clean build directories, record exact commands and logs, and run at least
the following before accepting the design.

### Linux staged mode

- Ninja default full build with high parallelism.
- Clean direct builds of `hipblaslt-test`, `hipblaslt-test-data`, a scoped device
  library, default ext-ops, and `all`.
- Repeat clean and no-op builds to expose missing dependencies.
- Rebuild after changing client sources, Python sources, package metadata, and
  ROCm version metadata.
- Delete the venv, client, individual staged byproducts, and stamp separately;
  verify reconstruction.

### Linux installed mode

- Version-matched wheel, client, rocisa, and ROCm root.
- Missing client, missing rocisa, wrong release, mismatched wheel, and malformed
  installation negative cases.
- Wheel-only `logic` and `create-library` smoke tests with no checkout on
  `sys.path`.

### Host-only compatibility

- Python 3.8 and 3.9 without Python development headers.
- No device libraries, client, or Python artifacts requested.

### Offline and environment behavior

- Network-disabled staged build.
- Clean installed build from a no-index wheelhouse.
- Minimal dependency environment and conflicting dependency versions.
- Verify imported package origins and run `pip check`.

### Multi-config and Windows

- Ninja Multi-Config Debug to Release to Debug, verifying staged client identity
  after each transition.
- Windows staged and installed device-library smoke tests.
- ROCm roots containing spaces.
- Nondefault install library and libexec settings.

### Integration and output stability

- Every shipped CMake preset, especially `hipblaslt-clients`.
- ASAN and TSAN code-generation paths.
- Coverage workflow using the intended instrumented rocisa.
- Representative generated metadata and kernel hashes compared with the parent
  commit.
- Standalone hipBLASLt, TheRock, packaging, and at least one downstream workflow.
- Shared CI lane evidence for affected supported architectures.

## Review questions

### What functionality changes?

The commit replaces checkout-relative Python execution and automatic in-tree
rocisa decisions with explicit staged-source and installed-package workflows.
It introduces a private editable TensileLite installation, a synthetic ROCm
artifact root, package-based logic and library commands, and configure-time
installed-package validation.

### What test level is appropriate?

This is build/integration infrastructure with platform and artifact-boundary
effects. Unit tests alone are insufficient. It needs CMake graph tests,
clean/incremental integration builds, installed-wheel tests, Linux and Windows
coverage, and at least one scoped device-library build on an affected GPU
architecture.

### Is omission of tests acceptable?

No. The commit changes a default shipping build path and supplies no automated
replacement coverage or documented waiver. The local implementation notes are
useful evidence but do not replace shared CI or executable regression tests.

### What adjacent tests should be considered?

- hipBLASLt test-data generation;
- every CMake preset;
- ext-op generation;
- device-library logic validation and compilation;
- client build and launch;
- rocisa coverage and source-development workflows;
- package wheel validation;
- host-only builds;
- downstream GEKO or equivalent integration;
- multi-config and Windows toolchain behavior.

## Evidence actually collected

The review performed:

- exact-commit and parent diff inspection;
- `git diff --check` successfully;
- Python AST parsing of the changed `tasks.py` successfully;
- JSON parsing of the changed `CMakePresets.json` successfully;
- inspection of the current CMake dependency edges and presets;
- a successful no-op build of the current
  `hipblaslt-tensilelite-python-env` target;
- inspection of the existing private venv, confirming that TensileLite is
  installed in the private environment while rocisa is imported from the parent
  environment through `hipblaslt-bootstrap.pth`.

The review did **not** execute:

- a clean build of exact commit `6382dcf61c`;
- a GPU device-library generation test;
- a Windows build;
- a multi-config build;
- shared CI.

Findings that depend on those executions are identified above as contract or
graph mismatches rather than claimed observed runtime failures.
