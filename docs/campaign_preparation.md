# Preparing a real campaign

`character-prepare` reads a campaign directory and writes the configuration a
character run needs, together with an audit of how every value was decided.

```bash
namd-analysis character-prepare \
  --shprop-dir /path/to/NuTest \
  --files SHPROP.37 SHPROP.171 SHPROP.425 SHPROP.848 SHPROP.1625 \
  --projection-dir /path/to/FAPI_001_BCF_PCBM_A \
  --preset bcf_pcbm \
  --campaign A \
  --frame-mode dish-cyclic \
  --cycle-length 1999 \
  --slurm-account loni_perovsk27 \
  --slurm-partition workq \
  --write-sbatch \
  --run-preflight
```

Generates:

```
state_map.json
atom_groups.json
projection_manifest.json
prepare_report.json
run_character_test.sbatch
```

`character-init` is the same command. `character-sbatch` is the same command
with `--write-sbatch` implied.

## Where the basis, the origin and the period come from

Production SHPROP files are plain numeric tables. `BMIN`/`BMAX` are properties
of the NAMD input and need not appear in one at all, and `NAMDTINI` often
survives only in the historical `SHPROP.<start-frame>` filename. Each quantity
has an ordered list of independent sources:

| quantity | precedence |
| --- | --- |
| exact VASP bands | `state_map.json: band_numbers` > registered campaign provenance > optional *agreeing* SHPROP `BMIN`/`BMAX` |
| `NAMDTINI` | optional SHPROP metadata > validated `SHPROP.<integer>` filename suffix |
| cyclic period | explicit `projection_manifest.json: cycle_length` > optional `NSW - 1` |

Three rules apply to all of them:

- **A present key must be readable.** `NAMDTINI = 37.5` is not the same as a
  missing `NAMDTINI`. Treating it as absent would silently demote to the
  filename and use a value the file itself contradicts, so it is refused.
- **Two sources that disagree are not reconciled.** A `NAMDTINI` header that
  contradicts the filename suffix is an error: choosing either would shift
  every frame assignment invisibly.
- **A higher source that wins is still compared with the lower one.** Bands are
  the one quantity nothing downstream can detect as wrong — a plausible
  population comes out either way — so when `band_numbers` or a preset supplies
  the basis, any `BMIN`/`BMAX` the files carry is still read and a disagreement
  is reported as a warning in `prepare_report.json`, `preflight` and the run.

For campaign A the basis is `[976, 977, 978, 979, 980, 981]` and
`state_map.json` records it explicitly with
`"band_numbers_source": "preset_bcf_pcbm_A"` — it is campaign provenance, not
SHPROP inference. Campaigns B and C are registered as *known but unresolved*:
asking for them is an error that says their provenance was never supplied, and
they cannot inherit A's.

## Three kinds of claim, never mixed

The command prints, and `prepare_report.json` records, a `source` for every
value:

| `source` | meaning |
| --- | --- |
| `inferred_from_shprop_table` | read out of the numbers, with the evidence recorded |
| `inferred_from_shprop_headers` | read out of `BMIN`/`BMAX`/`NAMDTINI`/`NSW` |
| `projection_directory_coverage` | read from which frame directories exist |
| `preset_<name>_<campaign>` | a declaration a human established for that campaign |
| `explicit_cli` | you said so |

A preset is **not** inference. It records what was true of the structure that
was actually built, and it is checked against the files before it is used: a
preset whose ion count disagrees with the representative PROCAR is an error,
not something rescaled to fit.

## What it refuses to do

- **Name a subsystem, or place an atom boundary.** Element symbols and
  contiguity are not evidence of physical partition. Without `--preset` or
  `--atom-groups` the command stops.
- **Pick between two readings of a table.** See below.
- **Infer the frame mode.** `--frame-mode` is required and is never guessed
  from filenames or directory layout.
- **Choose a cycle length when the two sources disagree.** See below.

## Inferring the population columns

The header gives the basis size `n = BMAX - BMIN + 1`. A candidate is one
contiguous block of `n` columns that excludes the time column. Each candidate
is then tested against real values from a bounded sample of rows:

- every value lies in `[0, 1]`;
- every row sums to one, within the same tolerance used everywhere else.

Exactly one candidate must survive. For the six-state BCF/PCBM layout — time,
an energy-like column, then six populations — there are two candidates,
`[1..6]` and `[2..7]`, and the first fails both tests because it includes the
energy column. That is evidence, not an assumption about where energies live.

If two candidates survive, the table genuinely does not say which is right and
the command stops. If none survive, it prints what each candidate did and why
it failed.

`prepare_report.json` records every candidate, accepted or not, under
`population_column_rationale`.

**The sample is bounded.** Candidates are tested against the first
`COLUMN_SAMPLE_ROWS` (2000) rows so that preparation costs the same on a 900 MB
history as on a small one. A file whose populations leave `[0,1]` or stop
summing to one only later in the trajectory is *not* caught here. It is caught
by the analysis, which validates every row of every chunk and refuses to produce
a number. The report states how many rows were sampled out of how many exist,
and `sample_covers_whole_file` says plainly whether the check was exhaustive.

**The column block is not the state identity.** Which state sits in which
column is a property of the run that wrote SHPROP; no preset-free run of this
command fills in `groups`, and one that cannot fill them says so in
`unresolved_questions` and exits non-zero.

## Discovering frames

A frame is a subdirectory whose name is entirely digits and which contains a
`PROCAR` (override with `--procar-name`).

- Numeric directories without a PROCAR are reported, not counted.
- Gaps between the first and last frame are reported; a history that visits a
  missing frame will fail preflight.
- **Two spellings of the same number** (`1` and `0001`) are an error. Which
  PROCAR would be read depends on directory order.

A regular zero-padded tree with no gaps gives a pattern manifest:

```json
{
  "procar_pattern": "/abs/path/FAPI_001_BCF_PCBM_A/{frame:04d}/PROCAR",
  "first_frame": 1,
  "last_frame": 1999,
  "frame_step": 1,
  "cycle_length": 1999
}
```

Anything irregular falls back to explicit per-frame entries. Both are valid
manifests; the pattern is just shorter.

## Proposing a cycle length

Two independent statements about the same number:

- **directory coverage** — frames `1..N` present with no gaps proposes `N`;
- **`NSW - 1`** from the SHPROP headers.

| situation | what happens |
| --- | --- |
| they agree | `cycle_length` is written, source `projection_directory_coverage` |
| the directory starts above 1 or has gaps | `NSW - 1` is used, source `inferred_from_shprop_headers` |
| **they disagree** | **nothing is chosen**; no `cycle_length` is written, the report says so, and the command exits 3 |
| histories imply different periods | same: nothing is chosen |
| `--cycle-length` given | it wins, and any disagreement is still recorded |

The engine that wrote the headers is the authority on its own bookkeeping, and
this package cannot see it. A disagreement means either frames are missing from
the archive or the header describes a different run length — both worth knowing
before the analysis, neither resolvable from the files.

## Memory

`character-prepare` passes these through to both steps of the generated script:

| flag | effect |
| --- | --- |
| `--memory-budget 24G` | ceiling checked against an estimate made before any file is opened; a run that cannot fit is refused rather than started |
| `--retain-per-file auto\|yes\|no` | keep the per-history projected populations. They are a diagnostic: the mean and SEM come from a running accumulator and never depend on them |
| `--accumulator-memmap-dir DIR` | spill the running mean/variance to memory maps |
| `--shprop-chunk-rows N`, `--shprop-io-mode` | how much of a history is resident at once |

`auto` estimates every array, prints what it will retain, stream or spill, and
says why. A malformed budget is rejected at preparation, not hours into a
queued job.

## The generated batch script

```
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=32G --time=04:00:00
```

Overridable with `--slurm-account`, `--slurm-partition`, `--slurm-memory`,
`--slurm-time`, `--slurm-cpus`, `--slurm-nodes`, `--slurm-tasks`. These are a
starting proposal, not a measurement.

The script sets `set -euo pipefail`, prints host, date, working directory, job
id, the resolved `namd-analysis`, the package version and the git commit when
running from an editable checkout, and lists its inputs. Then:

- it writes into a **job-specific** directory and **refuses to start** if a
  completed result (`report.json`) is already there, so a resubmission cannot
  quietly overwrite one;
- it puts the memory-mapped accumulators on `$SLURM_TMPDIR` when the scheduler
  provides it, falling back to `$TMPDIR` then `/tmp`;
- **preflight runs first** — a non-zero exit aborts the job before any PROCAR
  is parsed;
- the analysis runs under `/usr/bin/time -v`, and the maximum resident set size
  is repeated on its own line so a log skim finds it;
- scratch is removed on a clean exit and **kept, with a listing, on failure**,
  where what was half-written is worth the disk.

## Reading `prepare_report.json`

Beyond the provenance blocks it carries the SHPROP survey (rows, columns,
bytes, `BMIN`/`BMAX`, every `NAMDTINI` and `NSW`), the full column rationale,
the atom-group validation, the frame discovery summary with missing frames, the
representative PROCAR structure, the cycle-length decision with its note,
`warnings`, `unresolved_questions`, and `refusals` — the list of things the
command will not automate at all.

Exit codes: `0` prepared cleanly, `2` could not prepare, `3` prepared but
something is unresolved.
