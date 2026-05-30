# AGENTS.md

## Working principles

* Use the minimum number of tokens required to produce a correct, maintainable, and well-validated solution.
* Optimize for token efficiency without compromising:

  * correctness
  * robustness
  * maintainability
  * scientific validity
  * validation quality
  * architectural reasoning
* Keep responses concise and focused.
* Work in milestone mode, not micro-step mode.
* Batch related changes together instead of splitting work into many small steps.
* Avoid cosmetic refactors unless they improve:

  * correctness
  * maintainability
  * robustness
  * clarity
* Only propose follow-up work when it materially improves:

  * correctness
  * robustness
  * maintainability
  * test coverage
  * scientific validity

## Project context

This repository contains Clear Day Analysis.

The purpose of the project is to provide a reproducible workflow for:

* reading Typical Meteorological Year (TMY) datasets from multiple providers
* computing solar position information
* fitting clear-day DNI envelope models
* classifying days according to atmospheric clarity
* generating plots, reports, and derived solar resource metrics

The repository is intended to support solar resource assessment and CSP/solar-thermal project development.

## Project status context

Before starting any non-trivial task, read `PROJECT_STATUS.md`.

Use `PROJECT_STATUS.md` to understand:

* current priorities
* supported TMY formats
* active architectural decisions
* known technical debt
* pending validation
* recently completed milestones

Keep durable working rules in this file.

Keep time-sensitive project context in `PROJECT_STATUS.md`.

When a task materially changes the state of the project, update `PROJECT_STATUS.md` accordingly.

Examples include:

* addition of a new TMY source
* changes in datetime conventions
* major workflow changes
* new validation procedures
* completion of significant milestones
* resolution of technical debt items
* changes to project priorities

Do not update `PROJECT_STATUS.md` for trivial edits.

## Codex efficiency guidelines

* Read only the files necessary for the current task.
* Avoid repository-wide searches unless architectural impact is unclear.
* Reuse existing abstractions whenever possible.
* Prefer extending existing workflows over introducing parallel implementations.
* Avoid duplicated logic.
* Avoid creating multiple ways to perform the same task.
* Stop once the requested milestone has been completed.
* Do not perform speculative refactors.

## Architecture guidelines

### TMY ingestion

All TMY files should be loaded through the common reader interface.

The preferred entry point is:

```python
read_tmy_csv(...)
```

Source-specific readers should remain implementation details whenever practical.

Currently supported formats may evolve over time and should be documented in `PROJECT_STATUS.md`.

When adding support for a new TMY source:

* preserve a common DataFrame schema
* preserve common metadata fields
* avoid source-specific workflow branches elsewhere in the repository
* extend the common reader abstraction rather than creating parallel workflows

### Datetime policy

The repository uses normalized TMY calendars for analysis.

The standard UTC analysis timestamp column is:

```text
datetime
```

Requirements:

* monotonic increasing
* timezone-aware UTC
* suitable for annual TMY analysis
* suitable for solar-position calculations, clear-day fitting, exports, and row ordering

The standard daily grouping timestamp column is:

```text
tmy_datetime_local
```

Requirements:

* timezone-naive local standard time
* normalized to a fixed non-leap synthetic TMY year
* suitable for daily DNI integration, day classification, and day-based plots

Source-specific timestamp columns may be retained for traceability.

Examples:

```text
nsrdb_datetime_utc
solargis_datetime_utc
pvgis_datetime_utc
```

These source-specific timestamps are intended for:

* auditing
* debugging
* validation
* export

UTC-based analysis algorithms should use the normalized `datetime` column. Daily grouping and classification workflows should use `tmy_datetime_local`.

### Analysis workflow

The standard workflow is:

1. Read TMY data
2. Compute solar position
3. Fit the clear-day model
4. Compute daily DNI integrals
5. Compute daily clearness metrics
6. Classify days
7. Generate plots and reports

Avoid duplicating workflow logic across scripts.

When possible, reusable functionality should reside in library modules rather than scripts.

### Plotting

Plots should be based on:

* normalized TMY ordering
* month/day seasonal interpretation
* TMY day number

Plots should not depend on source-specific calendar years.

### Scientific integrity

Preserve the physical meaning of all meteorological and irradiance quantities.

Do not silently modify:

* irradiance units
* timezone conventions
* datetime conventions
* DNI definitions
* DHI definitions
* GHI definitions
* solar position conventions

Document assumptions explicitly whenever changes affect scientific interpretation.

## Validation requirements

When relevant:

* run pytest
* run any repository validation workflow already in use
* verify that existing supported TMY formats continue to load correctly
* verify that existing classification results remain stable unless a change is intentional

New TMY readers should be validated using representative real files whenever possible.

Changes affecting datetime handling must include validation of:

* timezone awareness
* monotonicity
* daily grouping
* annual ordering
* downstream classification behavior

## Output requirements

For each completed task:

* Provide a concise summary.
* List modified files.
* List validation performed.
* Provide one production-ready commit message.

Do not print full file contents unless explicitly requested.

Do not provide multiple alternative commit messages unless explicitly requested.

When a task reaches the requested objective, stop and wait for further instructions.

## Commit message requirements

Whenever a task results in a logical software change that should be committed, provide a commit message following:

https://www.conventionalcommits.org/en/v1.0.0/

Use the Conventional Commits structure:

```text
<type>[optional scope]: <description>

<body>

<footer>
```

Use scopes when they improve clarity.

Examples of valid commit types:

* feat
* fix
* refactor
* perf
* test
* docs
* build
* ci
* chore
* revert

Breaking changes must follow the Conventional Commits specification.

The commit message should explain:

* what changed
* why it changed
* important implementation decisions
* compatibility considerations
* validation performed

Do not merely restate modified filenames.

### Commit message formatting

Do not artificially wrap text.

Do not insert line breaks to fit terminal, GitHub, GitLab, or VS Code display widths.

Use line breaks only:

* between the subject and the body
* between body paragraphs
* before footers
* where required by normal language grammar

Each paragraph should be written as continuous prose regardless of length.

The generated commit message should be ready to use directly in Git without reformatting.

## PROJECT_STATUS.md maintenance

`PROJECT_STATUS.md` is the primary project-context document used to provide background and continuity across development sessions.

Before starting any non-trivial task:

1. Read `PROJECT_STATUS.md`.
2. Use it to understand the current state of the project.
3. Verify whether the information is still accurate.

When a task materially changes the project state:

* update `PROJECT_STATUS.md`
* keep it concise
* preserve historical context only when useful for future development decisions

Examples of changes that require updating `PROJECT_STATUS.md`:

* support for a new TMY source
* changes to the analysis workflow
* changes to datetime conventions
* major validation milestones
* significant technical debt discovered or resolved
* architectural changes
* new project priorities

### Initial creation

If `PROJECT_STATUS.md` does not exist:

1. Review the repository.
2. Infer the current project status.
3. Create a first proposed version of `PROJECT_STATUS.md`.
4. Present the proposed content for review.
5. Wait for approval before finalizing it.

The initial proposal should include:

* project purpose
* supported TMY formats
* current workflow architecture
* datetime conventions
* current priorities
* technical debt
* validation status
* next recommended milestones

### Updating behavior

When updating `PROJECT_STATUS.md`:

* preserve useful historical context
* remove obsolete information
* avoid duplication with `AGENTS.md`
* keep the document concise and actionable

`AGENTS.md` contains stable working rules.

`PROJECT_STATUS.md` contains the current state of the project.
