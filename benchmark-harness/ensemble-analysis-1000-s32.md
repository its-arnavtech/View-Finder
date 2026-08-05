# Ensemble and selective-agreement analysis

Rows: **1000**. Success definition: **≤200 km**.

This grid is exploratory and cannot itself be used as held-out calibration.

## Head-to-head

| Pair | Left closer | Right closer |
|---|---:|---:|
| GeoCLIP vs PLONK OSV-5M | 405 (40.5%) | 595 (59.5%) |
| GeoCLIP vs PLONK YFCC | 544 (54.4%) | 456 (45.6%) |
| PLONK OSV-5M vs PLONK YFCC | 630 (63.0%) | 370 (37.0%) |

## Oracle upper bound

This is not deployable; it shows how often at least one model was right.

- At least one model within 200 km: **249/1000 (24.9%)**

## Fixed-group agreement grid

| Models | Agreement radius | Accepted | Coverage | Successes | Observed | 95% Wilson lower |
|---|---:|---:|---:|---:|---:|---:|
| GeoCLIP + PLONK OSV-5M | 1 km | 0 | 0.0% | 0 | 0.0% | 0.0% |
| GeoCLIP + PLONK OSV-5M | 5 km | 0 | 0.0% | 0 | 0.0% | 0.0% |
| GeoCLIP + PLONK OSV-5M | 10 km | 0 | 0.0% | 0 | 0.0% | 0.0% |
| GeoCLIP + PLONK OSV-5M | 25 km | 7 | 0.7% | 4 | 57.1% | 25.0% |
| GeoCLIP + PLONK OSV-5M | 50 km | 14 | 1.4% | 6 | 42.9% | 21.4% |
| GeoCLIP + PLONK OSV-5M | 100 km | 44 | 4.4% | 14 | 31.8% | 20.0% |
| GeoCLIP + PLONK OSV-5M | 200 km | 105 | 10.5% | 38 | 36.2% | 27.6% |
| GeoCLIP + PLONK YFCC | 1 km | 0 | 0.0% | 0 | 0.0% | 0.0% |
| GeoCLIP + PLONK YFCC | 5 km | 0 | 0.0% | 0 | 0.0% | 0.0% |
| GeoCLIP + PLONK YFCC | 10 km | 1 | 0.1% | 1 | 100.0% | 20.7% |
| GeoCLIP + PLONK YFCC | 25 km | 4 | 0.4% | 3 | 75.0% | 30.1% |
| GeoCLIP + PLONK YFCC | 50 km | 12 | 1.2% | 4 | 33.3% | 13.8% |
| GeoCLIP + PLONK YFCC | 100 km | 30 | 3.0% | 10 | 33.3% | 19.2% |
| GeoCLIP + PLONK YFCC | 200 km | 82 | 8.2% | 26 | 31.7% | 22.6% |
| PLONK OSV-5M + PLONK YFCC | 1 km | 0 | 0.0% | 0 | 0.0% | 0.0% |
| PLONK OSV-5M + PLONK YFCC | 5 km | 0 | 0.0% | 0 | 0.0% | 0.0% |
| PLONK OSV-5M + PLONK YFCC | 10 km | 0 | 0.0% | 0 | 0.0% | 0.0% |
| PLONK OSV-5M + PLONK YFCC | 25 km | 4 | 0.4% | 2 | 50.0% | 15.0% |
| PLONK OSV-5M + PLONK YFCC | 50 km | 16 | 1.6% | 11 | 68.8% | 44.4% |
| PLONK OSV-5M + PLONK YFCC | 100 km | 39 | 3.9% | 22 | 56.4% | 41.0% |
| PLONK OSV-5M + PLONK YFCC | 200 km | 107 | 10.7% | 45 | 42.1% | 33.1% |
| GeoCLIP + PLONK OSV-5M + PLONK YFCC | 1 km | 0 | 0.0% | 0 | 0.0% | 0.0% |
| GeoCLIP + PLONK OSV-5M + PLONK YFCC | 5 km | 0 | 0.0% | 0 | 0.0% | 0.0% |
| GeoCLIP + PLONK OSV-5M + PLONK YFCC | 10 km | 0 | 0.0% | 0 | 0.0% | 0.0% |
| GeoCLIP + PLONK OSV-5M + PLONK YFCC | 25 km | 0 | 0.0% | 0 | 0.0% | 0.0% |
| GeoCLIP + PLONK OSV-5M + PLONK YFCC | 50 km | 2 | 0.2% | 1 | 50.0% | 9.5% |
| GeoCLIP + PLONK OSV-5M + PLONK YFCC | 100 km | 5 | 0.5% | 3 | 60.0% | 23.1% |
| GeoCLIP + PLONK OSV-5M + PLONK YFCC | 200 km | 29 | 2.9% | 14 | 48.3% | 31.4% |
