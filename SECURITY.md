# Security Policy

This project is a local-first benchmark toolkit. It should not upload prompts, API keys, benchmark artifacts, or endpoint credentials by default.

## Reporting a Security Issue

If you find a security issue, open a private security advisory if the hosting platform supports it. If private advisories are unavailable, contact the maintainer before filing a public issue with exploit details.

Useful reports include:

- affected command or module
- reproduction steps
- expected impact
- whether secrets, prompts, endpoint URLs, or benchmark artifacts can leak
- suggested fix, if known

## Benchmark and Data Safety

- Do not include API keys or private prompts in issues, logs, or sample artifacts.
- Use environment variable names, not secret values, in configs and reports.
- Treat remote endpoint URLs as sensitive unless they are localhost or explicitly intended for publication.
- Do not publish hardware performance claims without raw results, manifests, metadata, and validation output.

## Supported Versions

The project is pre-release. Security fixes target the current main development line until versioned releases are published.
