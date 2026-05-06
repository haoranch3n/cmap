#!/usr/bin/env bash
# Print one line per UMAP hyperparameter combo: n_neighbors min_dist metric
# Default 36 lines: nn {5,15,30} x md {0.0,0.1} x metric {cosine,euclidean}

set -euo pipefail

for nn in 5 15 30; do
  for md in 0.0 0.1; do
    for met in cosine euclidean; do
      echo "$nn $md $met"
    done
  done
done
