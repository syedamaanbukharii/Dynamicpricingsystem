# Contributing

Thanks for your interest in improving the AI-Powered Dynamic Hotel Pricing
System. This guide explains how to set up your environment, the standards we
follow, and how to propose changes.

## Development setup

Requires Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
# optional, for live scraping:
python -m playwright install chromium
```

## Standards

- **Type hints and docstrings** are required on all public functions, classes,
  and modules (Google-style docstrings).
- **No placeholders or dead code.** Every code path should be implemented and
  reachable. Optional integrations must degrade gracefully behind import guards.
- **Keep train/serve symmetry.** Features must flow through the shared
  `FeatureBuilder`, and `PricingFeatures.FEATURE_ORDER` is the single source of
  truth for the model's feature columns.
- **Pricing stays deterministic.** LLMs may be used for cleaning, normalization,
  and explanations — never for the numeric price decision.

## Quality gates

Before opening a pull request, ensure all of the following pass:

```bash
make format     # isort + black + ruff --fix
make lint       # ruff (no errors)
make typecheck  # mypy (no errors)
make test       # pytest (all green)
```

Or run everything at once:

```bash
make check
```

New behavior should come with tests. Aim to cover the happy path plus the
relevant edge cases (bounds, fallbacks, validation failures).

## Commit and PR guidelines

- Write clear, imperative commit messages (e.g. "Add occupancy-pressure rule").
- Keep pull requests focused; one logical change per PR where practical.
- Describe the motivation and summarize the approach in the PR description.
- Update `docs/` and `CHANGELOG.md` when you change behavior or interfaces.

## Reporting bugs and requesting features

Open an issue with:

- what you expected vs. what happened,
- minimal steps to reproduce (including environment and config),
- relevant logs (with secrets redacted).

## Security

Please do not file public issues for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for responsible-disclosure instructions.
