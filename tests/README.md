# Tests

Run from the repository root with the same interpreter that has CUDA PyTorch and Triton:

```bash
python -m unittest discover -s tests -v
```

| File | Scope | GPU |
|---|---|---|
| `test_contract.py` | Packaging layout, configuration validation, CLI exit codes and I/O | no |
| `test_pipeline_gpu.py` | Evidence hypothesis cases, blind two-signal recovery, alias suppression, single events, families, noise scales | yes |
| `test_evidence_kernels.py` | Dense float64 reference comparison of the surrogate and extended evidence kernels | yes |
| `test_survey.py` | Transit geometry derivation and CTOI row filling | no |
