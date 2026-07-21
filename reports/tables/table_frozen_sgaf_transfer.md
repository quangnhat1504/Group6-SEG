# Frozen A3 SGAF Transfer

Protocol: train the A3 gate on SciFact `trainfit`, freeze `C=0.1` and `coverage=0.05`, then apply the same gate to each target dataset without target-label tuning.

| Dataset | Stage | Method | Switch | nDCG@10 | Delta vs BGE-small | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| scifact | C:bm25 | Component bm25 | N/A | 0.6523 | -0.1665 | 0.7757 | 0.8731 | 0.6184 |
| scifact | C:bge_small | Component bge_small | N/A | 0.8188 | +0.0000 | 0.9349 | 0.9783 | 0.7875 |
| scifact | C:bge_base | Component bge_base | N/A | 0.7376 | -0.0812 | 0.8659 | 0.9700 | 0.7004 |
| scifact | O1 | Oracle component router | N/A | 0.8786 | +0.0598 | 0.9609 | 0.9783 | 0.8563 |
| scifact | A3-frozen | Frozen SciFact A3 SGAF | 0.0500 | 0.8216 | +0.0028 | 0.9382 | 0.9783 | 0.7900 |
| nfcorpus | C:bm25 | Component bm25 | N/A | 0.3079 | -0.0426 | 0.1524 | 0.2347 | 0.5085 |
| nfcorpus | C:bge_small | Component bge_small | N/A | 0.3505 | +0.0000 | 0.1710 | 0.3276 | 0.5385 |
| nfcorpus | C:bge_base | Component bge_base | N/A | 0.3695 | +0.0191 | 0.1756 | 0.3320 | 0.5655 |
| nfcorpus | O1 | Oracle component router | N/A | 0.4249 | +0.0744 | 0.2042 | 0.3381 | 0.6442 |
| nfcorpus | A3-frozen | Frozen SciFact A3 SGAF | 0.0495 | 0.3530 | +0.0025 | 0.1721 | 0.3277 | 0.5434 |
| fiqa | C:bm25 | Component bm25 | N/A | 0.2167 | -0.1468 | 0.2780 | 0.4737 | 0.2703 |
| fiqa | C:bge_small | Component bge_small | N/A | 0.3635 | +0.0000 | 0.4273 | 0.6544 | 0.4365 |
| fiqa | C:bge_base | Component bge_base | N/A | 0.3909 | +0.0274 | 0.4572 | 0.7314 | 0.4740 |
| fiqa | O1 | Oracle component router | N/A | 0.4650 | +0.1014 | 0.5252 | 0.6851 | 0.5615 |
| fiqa | A3-frozen | Frozen SciFact A3 SGAF | 0.0494 | 0.3653 | +0.0017 | 0.4284 | 0.6577 | 0.4427 |
| scidocs | C:bm25 | Component bm25 | N/A | 0.1495 | -0.0398 | 0.1543 | 0.3370 | 0.2681 |
| scidocs | C:bge_small | Component bge_small | N/A | 0.1893 | +0.0000 | 0.1996 | 0.4559 | 0.3250 |
| scidocs | C:bge_base | Component bge_base | N/A | 0.2147 | +0.0253 | 0.2278 | 0.5104 | 0.3533 |
| scidocs | O1 | Oracle component router | N/A | 0.2666 | +0.0773 | 0.2659 | 0.4749 | 0.4661 |
| scidocs | A3-frozen | Frozen SciFact A3 SGAF | 0.0500 | 0.1902 | +0.0008 | 0.2009 | 0.4571 | 0.3259 |
