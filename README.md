# R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning

[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](https://opensource.org/licenses/Apache-2.0)

This is an official JAX implementation of "[R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning](https://arxiv.org/abs/2605.14026)" to be presented at the Forty-Third International Conference on Machine Learning ([ICML 2026](https://icml.cc/virtual/2026/poster/62372)).

## Installation

* **Using Conda (Recommended):**
    ```bash
    conda env create -f deps/environment.yaml
    conda activate simba_v2
    ```
* **Other Methods:**
For other installation methods or detailed troubleshooting, please refer to the base repository: [SimbaV2](https://github.com/DAVIAN-Robotics/SimbaV2)

## Running Experiments
* **Single run** : Configuration should be modified before run. Configuration can be found at configs/online_rl.yaml
    ```bash
    python run_online.py
    ```
* **Parallel run** : Refer run_parallel.py to check arguments.
    ```bash
    python run_parallel.py --env_type dmc_hard --agent_config simbaV2_spl_r2r2 --num_seeds 5 --device_ids 0 1 2 3
    ```

## Acknowledgement
* This project is heavily based on the official codebase of [SimbaV2](https://github.com/DAVIAN-Robotics/SimbaV2). We express our deepest gratitude to the Davian Robotics team for open-sourcing their high-quality code.
* This project is licensed under the Apache License 2.0, adhering to the original license of SimbaV2.
