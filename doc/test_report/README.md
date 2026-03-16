# Test Report Directory

This directory contains comprehensive test reports for the PHMGA repository validation.

## Report Structure

```
doc/test_report/
├── README.md                    # This file
├── summary.md                   # Executive summary of test results
├── test_report.html             # Detailed pytest HTML report
├── coverage/                    # Coverage report directory
│   └── index.html              # HTML coverage report
└── artifacts/                   # Sample output artifacts from test runs
    ├── rm101_synth_dag/        # DAG generation artifacts
    ├── rm101_synth_ml/         # ML pipeline artifacts
    ├── rm101_synth_torch/      # PyTorch training artifacts
    └── ...                     # Other test artifacts
```

## Generating Test Reports

### Quick Start

```bash
# Generate all test reports
python scripts/generate_test_report.py

# Or manually generate individual reports
pytest --html=doc/test_report/test_report.html --self-contained-html
pytest --cov=src --cov-report=html:doc/test_report/coverage --cov-report=term
```

### Detailed Steps

1. **Run Test Suite**
   ```bash
   pytest -xvs --cov=src --cov-report=html:doc/test_report/coverage
   ```

2. **Generate HTML Report**
   ```bash
   pytest --html=doc/test_report/test_report.html --self-contained-html
   ```

3. **Generate Summary**
   ```bash
   python scripts/generate_test_report.py
   ```

4. **Collect Artifacts** (optional)
   ```bash
   cp -r artifacts/ doc/test_report/artifacts/
   ```

## Report Contents

### summary.md

Executive summary containing:
- Test results overview (passed/failed/skipped)
- Coverage metrics
- Production readiness assessment
- Squad-specific results
- Known issues and recommendations

### test_report.html

Detailed pytest HTML report containing:
- Individual test results
- Failure details and error messages
- Test duration and performance metrics
- Filterable by test status, duration, module

### coverage/index.html

HTML coverage report containing:
- Line-by-line coverage breakdown
- Per-module coverage statistics
- Missing coverage indicators
- Branch coverage (if enabled)

### artifacts/

Sample output artifacts from test runs:
- DAG JSON files
- Training curves and checkpoints
- Evaluation metrics
- Model build plans

## Production Readiness Criteria

The system is assessed as **production ready** when:

- [x] All P0 tests pass (≥95% pass rate)
- [x] All P1 tests pass (≥90% pass rate)
- [x] All end-to-end workflows complete successfully
- [x] No memory leaks or crashes
- [x] Documentation is accurate and complete
- [x] Known issues are documented (if any)

**Status Levels:**
- 🟢 **PRODUCTION READY** - All criteria met
- 🟡 **NEEDS IMPROVEMENT** - Some criteria not met, but functional
- 🔴 **NOT READY** - Critical failures or missing features

## Viewing Reports

### Local Viewing

Open the HTML reports in a web browser:

```bash
# Linux
xdg-open doc/test_report/test_report.html
xdg-open doc/test_report/coverage/index.html

# macOS
open doc/test_report/test_report.html
open doc/test_report/coverage/index.html

# Windows
start doc/test_report/test_report.html
start doc/test_report/coverage/index.html
```

### Sharing Reports

The HTML reports are self-contained and can be:
- Shared via email
- Hosted on a web server
- Included in documentation
- Attached to pull requests

## Continuous Integration

These reports can be automatically generated in CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pytest --cov=src --cov-report=html:doc/test_report/coverage
    pytest --html=doc/test_report/test_report.html --self-contained-html
    python scripts/generate_test_report.py

- name: Upload reports
  uses: actions/upload-artifact@v3
  with:
    name: test-reports
    path: doc/test_report/
```

## Historical Reports

To maintain historical test reports, organize them by date:

```bash
doc/test_report/
├── 2026-03-16/
│   ├── summary.md
│   ├── test_report.html
│   └── coverage/
├── 2026-03-17/
│   ├── summary.md
│   ├── test_report.html
│   └── coverage/
└── README.md
```

## Troubleshooting

### Missing pytest-html

```bash
pip install pytest-html
```

### Missing pytest-cov

```bash
pip install pytest-cov
```

### Permission Issues

```bash
chmod +x scripts/generate_test_report.py
```

### Coverage Report Not Generated

Ensure pytest-cov is installed and tests are run with `--cov` flag:

```bash
pytest --cov=src --cov-report=html:doc/test_report/coverage
```

## Additional Resources

- [Project README](../../README.md)
- [Testing Documentation](../../doc/structure/README.md)
- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
