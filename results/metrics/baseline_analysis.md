| Variant | Dim | Train acc. | Val acc. |
|---|---|---|---|
| All hand-crafted features (HOG + histograms + contact geometry) | 1792 | 1.000 | 0.161 |
| HOG only | 1764 | 1.000 | 0.165 |
| Contact statistics only (histogram, colour stats, geometry) | 28 | 0.538 | 0.122 |
| Control: raw 16x16 grey difference pixels | 256 | 0.822 | 0.113 |
| Leaky control: same features, random frame-level split (invalid protocol) | 1792 | 1.000 | 0.236 |
