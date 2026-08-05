### GeoCLIP accuracy

| Subset | n | <=1km | <=25km | <=200km | <=750km | <=2500km | Mean km | Median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 1000 | 0.0% | 1.2% | 9.4% | 34.9% | 64.9% | 3,560 | 1,374 |
| kaggle/4k-benchmark-images | 286 | 0.0% | 3.1% | 17.8% | 51.0% | 72.0% | 2,845 | 707 |
| kaggle/google-street-view | 714 | 0.0% | 0.4% | 6.0% | 28.4% | 62.0% | 3,847 | 1,687 |

### GeoCLIP confidence calibration

| Confidence bucket | n | Mean error km | Median error km |
|---|---:|---:|---:|
| 0-25% | 0 | – | – |
| 25-50% | 31 | 1,473 | 323 |
| 50-75% | 857 | 3,683 | 1,495 |
| 75-100% | 112 | 3,200 | 1,191 |

### Retrieval accuracy

| Subset | n | <=1km | <=25km | <=200km | <=750km | <=2500km | Mean km | Median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 1000 | 0.0% | 0.0% | 0.0% | 0.1% | 3.0% | 10,306 | 10,197 |
| kaggle/4k-benchmark-images | 286 | 0.0% | 0.0% | 0.0% | 0.3% | 4.9% | 10,276 | 10,590 |
| kaggle/google-street-view | 714 | 0.0% | 0.0% | 0.0% | 0.0% | 2.2% | 10,319 | 10,017 |

### Retrieval confidence calibration

| Confidence bucket | n | Mean error km | Median error km |
|---|---:|---:|---:|
| 0-25% | 0 | – | – |
| 25-50% | 0 | – | – |
| 50-75% | 0 | – | – |
| 75-100% | 1000 | 10,306 | 10,197 |

### PLONK OSV-5M accuracy

| Subset | n | <=1km | <=25km | <=200km | <=750km | <=2500km | Mean km | Median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 1000 | 0.0% | 0.6% | 14.4% | 42.3% | 69.7% | 2,954 | 996 |
| kaggle/4k-benchmark-images | 286 | 0.0% | 1.0% | 24.5% | 57.7% | 78.3% | 2,340 | 568 |
| kaggle/google-street-view | 714 | 0.0% | 0.4% | 10.4% | 36.1% | 66.2% | 3,201 | 1,245 |

### PLONK OSV-5M confidence calibration

| Confidence bucket | n | Mean error km | Median error km |
|---|---:|---:|---:|
| 0-25% | 0 | – | – |
| 25-50% | 0 | – | – |
| 50-75% | 1000 | 2,954 | 996 |
| 75-100% | 0 | – | – |

### PLONK YFCC accuracy

| Subset | n | <=1km | <=25km | <=200km | <=750km | <=2500km | Mean km | Median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 1000 | 0.0% | 0.6% | 9.0% | 27.8% | 51.6% | 4,773 | 2,263 |
| kaggle/4k-benchmark-images | 286 | 0.0% | 1.4% | 17.1% | 40.2% | 61.9% | 4,077 | 1,306 |
| kaggle/google-street-view | 714 | 0.0% | 0.3% | 5.7% | 22.8% | 47.5% | 5,052 | 2,796 |

### PLONK YFCC confidence calibration

| Confidence bucket | n | Mean error km | Median error km |
|---|---:|---:|---:|
| 0-25% | 0 | – | – |
| 25-50% | 0 | – | – |
| 50-75% | 1000 | 4,773 | 2,263 |
| 75-100% | 0 | – | – |

### Retrieval: error by top_similarity (raw match quality)

| top_similarity | n | Mean error km | Median error km |
|---|---:|---:|---:|
| 0.0-0.5 | 2 | 11,593 | 11,593 |
| 0.5-0.6 | 106 | 10,401 | 10,711 |
| 0.6-0.7 | 641 | 10,252 | 10,075 |
| 0.7-0.8 | 251 | 10,395 | 10,331 |
| 0.8-1.0 | 0 | – | – |

### Head-to-head

- Images scored by both predictors: **1000**
- GeoCLIP closer: **901** (90.1%)
- Retrieval closer: **99** (9.9%)
