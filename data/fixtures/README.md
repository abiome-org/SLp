# Synthetic OMF fixtures

Only tiny generated fixtures may be committed below this directory. They exist
to test contracts and contain no biological measurements. Real datasets and
derived arrays remain untracked OMF artifacts.

`slp11-world-smoke/generate.py` deterministically rebuilds the three synthetic
two-species snapshots used by `workloads/slp-1-1-world-smoke.yaml`. On a
supported Linux host, run:

```console
python3 data/fixtures/slp11-world-smoke/generate.py
omf --actor slp-researcher data add data/fixtures/slp11-world-smoke/pretrain --name slp-1-1-world-smoke-pretrain --rights rights/fixture-cc0.yaml
omf --actor slp-researcher data add data/fixtures/slp11-world-smoke/validation --name slp-1-1-world-smoke-validation --rights rights/fixture-cc0.yaml
omf --actor slp-researcher data add data/fixtures/slp11-world-smoke/reward --name slp-1-1-world-smoke-reward --rights rights/fixture-cc0.yaml
omf --actor slp-researcher data verify slp-1-1-world-smoke-pretrain
omf --actor slp-researcher data verify slp-1-1-world-smoke-validation
omf --actor slp-researcher data verify slp-1-1-world-smoke-reward
omf --actor slp-researcher run workloads/slp-1-1-world-smoke.yaml --binding bindings/local-linux.yaml
```
