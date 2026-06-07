# Why IVFPQ Was Removed

IVFPQ is no longer part of Copenhagen.

It was removed completely because, in this repository, it underperformed on the
only dimensions that mattered:

- Recall was materially worse than the TurboQuant path.
- Memory was not lower, because Copenhagen's old IVFPQ branch still retained the
  original float32 vectors for reranking, deletes, and split/rebalance logic.
- Complexity was higher, because the codebase had to maintain a second
  quantized-search implementation with separate training, encoding, compaction,
  and search logic.

## Repo-specific evidence

From the compression comparison in [README.md](README.md):

| Index | recall@10 | bytes/vec | compression |
|---|---:|---:|---:|
| Copenhagen float | 0.9971 | 512 | 1.0x |
| Copenhagen IVFPQ (old) | 0.6463 | 528 | 0.97x |
| TurboVec 4-bit | 0.8476 | 68 | 7.5x |
| TurboVec 2-bit | 0.6266 | 36 | 14.2x |
| Copenhagen-TQ block VQ `B=2` | 0.9070 | 72 | 7.1x |
| Copenhagen-TQ block VQ `B=4` | 0.7002 | 40 | 12.8x |

That old IVFPQ line is the key failure:

- `528 B/vector` is worse than the `512 B/vector` float baseline.
- `0.6463 recall@10` is far below the TurboQuant-family results at a fraction of
  the memory footprint.

So IVFPQ was failing both as a compression scheme and as a recall-preserving
approximation scheme.

## Why it failed here

The old path in `src/dynamic_ivf.cpp` used PQ codes only for coarse candidate
scoring, but it still kept full vectors around:

- Exact rerank needed the original vectors.
- Dynamic splits/rebalancing worked over the original vectors.
- Deletes and tombstone compaction still operated on the live float storage.

That meant we paid for:

- float32 storage
- PQ code storage
- extra training/encoding/search machinery

without getting the usual IVFPQ memory upside.

## Replacement

The compressed-search mode is now:

- `quant="tq"` in the public API
- TurboQuant-backed scoring in `src/dynamic_ivf.cpp`

## Follow-on work

The remaining quantization roadmap is not "bring IVFPQ back". It is:

- improve the TurboQuant path
- integrate block VQ / E8 / OPQ style extensions
- keep the public API strictly on `quant="tq"`
