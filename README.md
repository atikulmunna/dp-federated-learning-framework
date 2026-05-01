# DP-FedAvg: Federated Learning with Differential Privacy

A research prototype for studying the privacy-utility tradeoff in federated learning using DP-FedAvg, configurable Gaussian-mechanism noise, RDP privacy accounting, and dashboard-based experiment analysis.

## Overview

This repository is intended to implement an end-to-end federated learning workflow for comparing non-private FedAvg with differentially private federated training. The primary mechanism is central client-level DP-FedAvg: client updates are clipped, averaged by the server, and perturbed with Gaussian noise while a Rényi Differential Privacy accountant tracks cumulative privacy spend.

The project is designed around reproducible experiments rather than production deployment. Its focus is methodological clarity: defining the privacy unit, logging the accounting trace, separating sample-level and client-level privacy claims, and visualizing the resulting accuracy-privacy tradeoff.

## Planned Capabilities

- Federated learning simulation with Flower
- Baseline FedAvg training on MNIST and CIFAR-10
- DP-FedAvg with per-client update clipping and server-side Gaussian noise
- Optional per-client DP-SGD comparison using Opacus
- RDP accounting with conversion to `(epsilon, delta)` guarantees
- IID and Dirichlet non-IID client partitioning
- Multi-seed experiment runs with reproducible config files
- Privacy-utility visualization through a Streamlit dashboard

## Methodology

The primary privacy claim is client-level differential privacy under a trusted-aggregator setting. The intended DP-FedAvg workflow is:

1. Sample a client cohort for each round.
2. Train selected clients locally.
3. Compute each client's model update.
4. Clip each update to a fixed L2 norm.
5. Average clipped updates uniformly over the cohort.
6. Add Gaussian noise to the aggregate.
7. Step the privacy accountant with the round's noise multiplier and sampling rate.

Reported epsilon values are intended to represent cumulative privacy spend at the end of training, evaluated at a fixed delta.

## Repository Status

This repository is currently being initialized from the project specification. Implementation code, experiment configs, tests, and dashboard components will be added incrementally.

## Proposed Structure

```text
fl-dp/
├── configs/              # Experiment configurations
├── docs/                 # Public technical documentation
├── src/fldp/
│   ├── data/             # Dataset loading and client partitioning
│   ├── models/           # Small CNN models for experiments
│   ├── privacy/          # RDP accounting and inverse epsilon routines
│   ├── strategies/       # FedAvg, DP-FedAvg, and DP-SGD strategies
│   ├── train/            # Flower simulation driver
│   └── dashboard/        # Streamlit dashboard
├── tests/                # Unit and integration tests
└── runs/                 # Local run artifacts, ignored by git
```

## Privacy Notes

This project distinguishes between:

- **Client-level DP** for the primary DP-FedAvg mechanism
- **Sample-level DP** for any Opacus-based per-client DP-SGD comparison

These guarantees protect different privacy units and should not be reported interchangeably. Any released result should include the privacy unit, epsilon, delta, accountant method, sampling rate, number of rounds, clip norm, and noise multiplier.

## Development Goals

- Keep experiments reproducible from configuration files and seeds
- Validate privacy accounting against independent references
- Avoid test-set leakage during hyperparameter selection
- Preserve a clear boundary between public code, local specifications, and generated run artifacts

## License

No license has been selected yet. Until a license is added, all rights are reserved by the repository owner.
