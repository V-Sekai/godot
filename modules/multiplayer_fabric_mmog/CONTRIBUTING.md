# Contributing — multiplayer_fabric_mmog

This module is the MMOG layer sitting on top of `multiplayer_fabric`. It
targets the V-Sekai reference workload, so most real-world test content
and server code live outside this directory in sibling projects vendored
under `thirdparty/`.

## Companion projects

See [COMPANION_PROJECTS.md](../../COMPANION_PROJECTS.md) for the full
list of sibling modules and vendored projects that must stay in sync.

## Simplify, Then Add Lightness

Zack Anderson's application of Colin Chapman's racecar design philosophy to
hardware engineering. Development speed comes from reducing the mass of the
learning loop — ruthlessly deleting unnecessary requirements and pushing
complexity into software. When a program feels slow, don't ask how to go
faster; ask what unnecessary burdens can be dropped.

### Core lessons

- **Question and subtract requirements.** Examine which parts of a spec are
  not absolutely necessary.
- **Sequence your risks.** Early prototypes are scientific experiments
  designed to retire specific risks in order, not prove everything at once.
- **Insource the uncertain.** Mature components can be outsourced; core
  uncertainties stay in-house.
- **Shift complexity into software.** Replace physical complexity with
  computation.
- **Compress learning loops.** Distance between engineer and product is a
  tax on speed.
- **Maintain organizational lightness.** Small enough to naturally share
  context.

## Canonical references

- Quantitative facts (byte offsets, constants, command IDs) live in this
  module's C++ headers. If a value differs between the headers and
  `CONCEPT_MMOG.md`, the headers win.
- The simulation-side concept doc is
  `modules/multiplayer_fabric/CONCEPT_FABRIC.md`.
- Doc units are Hz, seconds, meters. "Tick" only appears in wire-field
  names and code identifiers, never in prose.
