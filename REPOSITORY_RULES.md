# Repository rules

Adapted for sphaira-rs from the tensor4all shared agent rules
(<https://github.com/tensor4all/tensor4all-agent-rules/>, `rules/common/*` and `rules/rust/*`).
Where this file is more specific it wins; for anything not covered, the shared rules apply.
The clean-room rules in [`AGENTS.md`](AGENTS.md) override everything here.

## Source of truth

- The code, its tests and the CI configuration define current behavior. When a document says something the code no
  longer does, treat the document as stale and fix it; never edit code to match stale prose.
- The physics and conventions come from the specification book at the commit in `SPEC_VERSION`. A gap or contradiction
  in the book is filed as an issue in the book repository, not resolved by guessing.

## Build profiles

- No default profile carries debug info: `dev`, `test`, `release`, `bench` and `ci` all set `debug = 0`
  (`debug = 1` and `"line-tables-only"` are debug info too). DWARF multiplies `target/` and the shared compiler cache
  (kache) for something no test reads.
- When symbols are needed, use `CARGO_PROFILE_DEV_DEBUG=2` for one session or the opt-in `release-debug` profile.
- CI uses `[profile.ci]` (inherits `test`, `incremental = false`, `strip = "symbols"`); workflows run
  `cargo test --profile ci`. Debug assertions and overflow checks stay on in `dev`, `test` and `ci`.
- `target/` is derived data. Clean only the current worktree's `target/`, never while a build runs there, and never
  delete the kache store; trim it with `kache gc`.

## Layering and public surface

- Crates follow the dependency order of the SCF (`comm` → `atomdata` → `structure` → `blocks`/`grid` → `twocenter` → `scf` → `cli`).
  No crate reaches into a lower crate's internals; fix behavior in the owning crate.
- Public APIs are contracts: keep them small and deliberate. Early development: remove or rename obsolete items directly,
  no compatibility shims, no `try_*` twins beside a panicking API.
- Declare shared external dependencies once in `[workspace.dependencies]` when two or more crates use them.
- Mutation is visible in names: unsuffixed returns an owned result, `*_mut` mutates in place, `*_into` overwrites a
  caller-provided output without reading it; accumulation uses `*_add_into`.

## Parallel and numerical conventions specific to sphaira-rs

- Physics kernels never communicate; drivers communicate only through `Comm` and precomputed plans
  (book: `design/data-structures.md` §2).
- Reductions are ordered (fixed order within a rank, increasing rank across ranks). No unordered MPI reductions,
  no atomics or arrival-order-dependent sums in anything that feeds a result. Same rank count ⇒ bitwise-identical result.
- Dense matrices are column-major. Grid fields are row-major `(i1, i2, i3)` (pencil-rs). Conversions are explicit, named functions.
- Dense linear algebra (GEMM, eigensolvers, factorizations) goes through tenferro-rs. Do not hand-write a
  contraction or factorization loop; if one is genuinely unavoidable, mark it `// contraction-kernel-ok: <reason>` and say why in the PR.
- A parallel split must keep working as the rank or thread count grows: no per-task reads of a shared large operand,
  no `O(P)` partial buffers without a bound, no fixed task caps; ranks with no work must be handled.

## Errors

- Public APIs return crate-local typed errors (`thiserror`), never `String`; preserve `source()` across crates.
- Classify failures by kind: input/format validation, unsupported feature, numerical failure (singular S, non-convergence),
  I/O, internal invariant. When one remedy exists, the message reads `<what failed>; <what to do>`.
- Every public function returning `Result` has a `# Errors` section naming the observable failure conditions;
  intentional panics have `# Panics`.

## Unsafe, invariants, hygiene

- `unsafe` is denied workspace-wide. A crate that needs it (FFI only) relaxes the lint locally with a written reason,
  and every `unsafe` block carries `// SAFETY:` naming the validation site.
- Non-obvious invariants kept by design (bounded loops, intentional copies, semantic zero-fill) are marked
  `// INVARIANT: <why it is valid, with the bound or validation site>`. An `#[allow(...)]` needs an adjacent `// INVARIANT:`.
- Public types implement `Debug`; hand-write a summary for types holding large buffers.
- Do not zero-fill buffers that are fully overwritten; do not allocate, clone or format inside per-grid-point or per-block hot loops.

## Tests

- Tests are small and fast: a few atoms, coarse grids, fixed seeds, tight or exact assertions; each test well under a second.
  SCF regressions and anything slower are marked heavy (`#[ignore = "heavy"]`) and run by a scheduled workflow, not on every PR.
- Production files hold production code. Unit tests live in `src/<module>/tests/*.rs` (with `#[cfg(test)] mod tests;`
  in the module), split by concern; crate-root `tests/` is for integration tests. A tiny leaf module may keep a small inline test.
- Prefer known values, analytic results, algebraic identities and physical invariants over shape or finiteness checks.
  Approximate comparisons report the residual (max absolute error, relative error).
- Distributed code is tested with `SimComm` for several rank counts (including more ranks than items) and for bitwise repeatability.
- Never relax a tolerance, skip a test or remove a check without a stated reason in the PR.

## Documentation

- Every public item has rustdoc naming the book section it implements, and a runnable example (`/// # Examples`)
  where an example teaches something; never `ignore` or `no_run`.
- Examples demonstrate real usage with meaningful assertions, not just that the code compiles.

## Pull requests

- One milestone issue (or a part of it) per PR; the body names the issue, the book commit, the spec sections, and a
  `Model: <model id>` line for AI-written changes.
- Local checks before a PR: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and the
  tests of the changed crates. Comprehensive runs belong to CI.
- Keep files focused; ~1000 lines is a review trigger, not a limit. Do not split a concept into many tiny files.
- When a PR changes durable design intent, update the book (via an issue there) in step.
