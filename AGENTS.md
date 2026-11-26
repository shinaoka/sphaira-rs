# Rules for agents and contributors

## Clean room (mandatory)

sphaira-rs is an **implementer** repository in the sense of the
[clean-room procedure](https://github.com/shinaoka/ab-initio-common-rust-substrate/blob/main/src/process/clean-room.md).

- Implement only from the specification book (at the commit in `SPEC_VERSION`), public documents
  (papers, OpenMX technical notes and manual) and fixtures.
- Do **not** read, clone, search or quote the source code of OpenMX or any derived version
  (including the GPU versions), and do not ask other agents for its contents.
- If the specification is insufficient, open an issue in the specification repository. Do not inspect the reference code.
- Record in each pull request the specification commit it was built against.

## Repository rules

Read [`REPOSITORY_RULES.md`](REPOSITORY_RULES.md) (build profiles with `debug = 0`, layering, ordered reductions,
errors, test layout, documentation, PRs). It adapts the tensor4all shared agent rules
(<https://github.com/tensor4all/tensor4all-agent-rules/>), which apply for anything it does not cover.
Every AI-written issue, issue comment and pull request discloses the model that wrote it
(`rules/common/provenance.md`).

