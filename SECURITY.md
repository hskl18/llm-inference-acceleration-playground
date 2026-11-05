# Security Policy

This project is a local-first benchmark toolkit. It should not upload prompts, API keys, benchmark artifacts, or endpoint credentials by default.

## Reporting a Security Issue

Report security issues privately through [GitHub private vulnerability reporting](https://github.com/hskl18/llm-inference-acceleration-playground/security/advisories/new).

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

Security fixes target the latest release.
