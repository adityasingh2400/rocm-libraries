# Contributing to rocke

Conventions and automation for the rocke engine
([platform/](platform/) + [library/](library/)). Pairs with the process map in
[library/ENGINEERING_PROCESS.md](library/ENGINEERING_PROCESS.md) and the hard
rules in [platform/AGENTS.md](platform/AGENTS.md).

---

## Table of Contents

- [Commit conventions](#commit-conventions)
- [Branch naming](#branch-naming)
- [Pre-commit hooks](#pre-commit-hooks)
- [Definition of Done gates](#definition-of-done-gates)

---

## Commit conventions

rocke uses [Conventional Commits](https://www.conventionalcommits.org/) with a
**required scope**, matching existing history (`feat(rocke):`, `test(rocke):`,
`fix(rocke):`, `perf(attention):`, `feat(rocke/attention):`, `docs(rocke):`).

```
<type>(<scope>): <subject>
```

- **type** — one of `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`,
  `ci`, `build`, `style`.
- **scope** — required. Use `rocke`, `hip-kernel-provider`, or a sub-area like
  `rocke/attention`.
- **subject** — imperative, lower-case, no trailing period.

Examples:

```
feat(rocke): add gfx950 mask-phase-split 2D lever
test(rocke/attention): add on-GPU numeric parity for d128 ring
fix(rocke): correct i64 paged-KV offset for >2GiB caches
```

The [check_commit_msg.py](tools/hooks/check_commit_msg.py) hook enforces this.
Merge/revert/fixup commits are exempt.

## Branch naming

`users/<github-username>/<kebab-branch-name>` (e.g. `users/anarao/gfx950-mask-split`).

## Pre-commit hooks

rocke provides four hook scripts under [tools/hooks/](tools/hooks/):

| Script | What it does |
|---|---|
| [guard_local_only.py](tools/hooks/guard_local_only.py) | blocks committing CLAUDE.md / SANDBOX_NOTES.md / PROBLEM.md / `.claude/` |
| [guard_artifacts.py](tools/hooks/guard_artifacts.py) | blocks committing `*.a`/`*.o`/`*.so`/`*.hsaco`/egg-info/`__pycache__`/`build/` |
| [check_commit_msg.py](tools/hooks/check_commit_msg.py) | enforces Conventional-Commit messages (`commit-msg` stage) |
| [remind_run_checks.py](tools/hooks/remind_run_checks.py) | **non-blocking** — reminds you to run `run_checks.py` before a PR |

Like other components (e.g. `gtest-format-miopen`), these run once their entries
are added to the monorepo root
[`.pre-commit-config.yaml`](../../../.pre-commit-config.yaml) under `- repo: local`
— a small follow-up. The scripts are runnable standalone in the meantime.

## Definition of Done gates

These heavier checks build the C++ engine and take minutes, so they are **not**
git hooks — run [`tools/run_checks.py`](tools/run_checks.py) on demand before
opening a PR (the `rocke-tests-reminder` hook nudges you to). See
[library/ENGINEERING_PROCESS.md](library/ENGINEERING_PROCESS.md) and
[DEFINITION_OF_DONE.md](DEFINITION_OF_DONE.md) for the full checklist. The stages:

1. Relative-path guard.
2. **Byte-identity gate at both llvm20 and llvm22.**
3. **Library parity** ([run_library_parity.py](library/tests/parity/run_library_parity.py)) —
   the `.py`/`.c` emitter pairs for all library operator families (attention,
   `fmha_*`, sage, sparse, the `gfx9xx_attention_tiled_*` variants) that the
   platform gate does not cover.
4. pytest (`platform/tests library/tests`).
5. On-GPU numeric parity — needs a HIP device; the lane that catches correctness
   regressions a spec-only test would miss. Only exercises the arch of the GPU on
   your machine, so run it on each arch you can.
