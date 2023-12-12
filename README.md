# READ-ZAP: Report
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This module compiles and evaluates the results from [the READ-ZAP amplicon-based haplotype phasing pipeline](https://github.com/nrminor/READ-ZAP). In particular, it reports the haplotypes at each amplicon sorted by the frequencies, alongside the synonymous and nonsynonymous mutations in those haplotypes. Eventually, this script will be integrated into the pipeline, but for now, we've placed it in its own repository and development environment.

Example usage:
```
usage: read_zap_report.py [-h] --results_dir RESULTS_DIR [--config CONFIG]

options:
  -h, --help            show this help message and exit
  --results_dir RESULTS_DIR, -d RESULTS_DIR
                        Results 'root' directory to traverse in search if iVar tables and other files..
  --config CONFIG, -c CONFIG
                        YAML file used to configure module such that it avoids harcoding.
```

To set up this script, we recommend users clone this repository, and, [with Poetry installed](https://python-poetry.org/), run `poetry install` to create the module environment. Then, the module can be run with the simpler `rz_report -d results/` command.
