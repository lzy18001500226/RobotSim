# ADR-0001: Docker is the reproducible environment

Status: Accepted

## Decision
Use Dockerfiles and Compose as the environment source of truth.

WSL2 is a Windows development host, not the reproducibility artifact.

## Consequences
- Host remains clean.
- Native Ubuntu and WSL2 use the same image.
- Heavy ML dependencies can be separated into dedicated containers.
