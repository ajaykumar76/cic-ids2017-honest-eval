\# How Much Do Random Splits Inflate ML Intrusion Detection Results? An Honest Evaluation on CIC-IDS2017



A random forest scores \*\*F1 = 0.9993\*\* on CIC-IDS2017 with a standard random train/test split. Tested on attack families it has never seen, the same approach scores \*\*F1 = 0.54\*\*. This project measures that gap, compares random forest and XGBoost under three evaluation setups, and uses SHAP and feature ablation to check what the models actually rely on.



\## Key findings



1\. \*\*Random splits inflate results.\*\* F1 falls from 0.9993 (random split) to 0.54 (train Mon-Wed, test Thu-Fri).

2\. \*\*Detecting new attack types is hard.\*\* On unseen families, recall was 37-38%. DDoS was partly caught (64%), while PortScan (0.4%), Bot (0%) and Infiltration (0%) were almost entirely missed.

3\. \*\*Known attacks in later traffic are detected well.\*\* With an ordered split within each file, XGBoost reached F1 = 0.9906, though Heartbleed was missed entirely.

4\. \*\*The model leans on lab-specific features.\*\* `Destination Port` and TCP window features top the SHAP ranking. Removing them barely changes recall but raises the false positive rate about 20x (0.135% to 2.6%).



\## Dataset and cleaning



CIC-IDS2017 (Canadian Institute for Cybersecurity, UNB), `MachineLearningCSV` version: 8 CSV files of labeled network flows, captured over five days. Download from the \[official page](https://www.unb.ca/cic/datasets/ids-2017.html).



| Step | Rows affected |

|---|---|

| Rows loaded | 2,830,743 |

| Removed: inf/NaN values | 2,867 |

| Removed: duplicate rows | 329,691 (11.6%) |

| Dropped: constant columns | 8 columns |

| \*\*Final dataset\*\* | \*\*2,498,185 rows, 70 features\*\* |



The classes are heavily imbalanced: benign traffic is 83% of flows, while Infiltration (36), Web Attack SQL Injection (21) and Heartbleed (11) have almost no samples.



!\[Flows per label](reports/label\_counts.png)



\## Experiments



All models were trained with each class capped at 50,000 rows to keep training fast. Test sets are never sampled or capped, except in Experiment A.



| Experiment | Train | Test | Purpose |

|---|---|---|---|

| A. Random split (baseline) | 70% random | 30% random | The naive, commonly reported setup |

| B. By day | Monday-Wednesday | Thursday-Friday | Detecting unseen attack families |

| C. Ordered within files | first 70% of each file | last 30% of each file | Known attacks, later traffic, no shuffling |



\### Results



| Experiment | Model | Precision | Recall | F1 | False positive rate |

|---|---|---|---|---|---|

| A. Random split\* | Random forest | 0.9996 | 0.9990 | \*\*0.9993\*\* | 0.15% |

| B. By day | Random forest | 0.9895 | 0.3680 | 0.5364 | 0.11% |

| B. By day | XGBoost | 0.9945 | 0.3774 | 0.5472 | 0.06% |

| B. By day | XGBoost (weighted) | 0.9950 | 0.3777 | 0.5476 | 0.05% |

| C. Ordered | Random forest | 0.9864 | 0.9556 | 0.9708 | 0.13% |

| C. Ordered | XGBoost | 0.9861 | 0.9950 | \*\*0.9906\*\* | 0.14% |

| C. Ordered | XGBoost (weighted) | 0.9875 | 0.9749 | 0.9811 | 0.12% |



\\\*Experiment A uses a capped sample (234,187 rows, of which 79% are attacks), so its false positive rate is not comparable to B and C, which use the full test data.



XGBoost matched or beat random forest in both setups. Class weighting did not help. These are single runs with a fixed seed, so differences of a point or two should not be over-interpreted.



\### Per-attack recall (random forest)



| Attack | B. By day | C. Ordered |

|---|---|---|

| BENIGN (false positive rate) | 0.11% | 0.13% |

| DDoS | 63.6% | 99.9% |

| PortScan | 0.4% | 99.8% |

| SSH-Patator | not in test | 99.8% |

| Bot | 0.0% | 76.5% |

| DoS GoldenEye | not in test | 70.3% |

| Web Attack, Brute Force | 9.3% | not in test |

| Web Attack, XSS | 8.1% | not in test |

| Web Attack, SQL Injection (21 flows) | 52.4% | not in test |

| Infiltration (36 flows in B, 4 in C) | 0.0% | 50.0% |

| Heartbleed (11 flows) | not in test | 0.0% |



Attack types appear in contiguous blocks, so Experiments B and C each leave out different attacks. Compare rows only where both columns have a value.



\## What does the model rely on?



!\[SHAP summary](reports/shap\_summary.png)



The SHAP summary (XGBoost, Experiment C) ranks `Destination Port` first, followed by `Init\_Win\_bytes\_forward`, `Bwd Packet Length Mean`, `Bwd Packet Length Min` and `Init\_Win\_bytes\_backward`. Destination port and TCP window settings can reflect how the attack tools and machines were set up in the lab, not general attack behavior, so I tested how much the model depends on them.



\### Ablation (XGBoost)



| Split | Features removed | Precision | Recall | F1 | False positive rate |

|---|---|---|---|---|---|

| C. Ordered | none | 0.9861 | 0.9950 | 0.9906 | 0.135% |

| C. Ordered | Destination Port | 0.9664 | 0.9628 | 0.9646 | 0.32% |

| C. Ordered | Port, window sizes, min\_seg\_size | 0.7851 | 0.9934 | 0.8771 | 2.63% |

| B. By day | none | 0.9945 | 0.3774 | 0.5472 | 0.06% |

| B. By day | Destination Port | 0.9929 | 0.3101 | 0.4727 | 0.06% |

| B. By day | Port, window sizes, min\_seg\_size | 0.9640 | 0.3764 | 0.5414 | 0.40% |



\*\*Interpretation.\*\* Without these features the model still detects nearly all known attacks (recall 99.3%), so it is also using genuine flow behavior. But the lab-specific features are what keep false alarms low: removing them raises the false positive rate about 20x, which would be unusable in a real SOC. On unseen attacks, the effect is mixed and not monotonic. Correlated features can substitute for one another, so this ablation likely understates the true dependence.



\## Limitations



\- \*\*Row order is a proxy for time.\*\* This CSV version has no timestamps. Experiment C assumes rows are roughly chronological within each file. The label distribution supports blocks of contiguous attacks, but I could not verify true chronological order.

\- \*\*Lab-generated traffic.\*\* Results may not transfer to real networks. No cross-dataset test has been run yet.

\- \*\*Single runs, one seed.\*\* Small differences between models are not statistically established.

\- \*\*Rare classes.\*\* Infiltration, SQL Injection and Heartbleed have very few samples, so their per-class numbers are unreliable.

\- \*\*Unseen-attack test is one specific split.\*\* Attack families differ in difficulty, so Experiment B does not measure detection of new attacks in general.



\## Future work



\- Repeat with multiple random seeds and report mean and spread

\- Cross-dataset evaluation on UNSW-NB15

\- Compare against an unsupervised anomaly detector for unseen attacks

\- Recompute the dataset with timestamps for a true time-based split



\## How to run



```bash

python -m venv venv

\# Windows: venv\\Scripts\\activate    Linux/Mac: source venv/bin/activate

pip install -r requirements.txt

```



1\. Download `MachineLearningCSV.zip` from the link above and extract the 8 CSVs into `data/raw/`.

2\. Run the scripts in order:



| Script | What it does |

|---|---|

| `src/01\_clean.py` | Loads, cleans, and saves `data/processed/cic2017\_clean.parquet` |

| `src/02\_explore.py` | Fixes label encoding, prints label statistics, saves the label chart |

| `src/03\_baseline.py` | Experiment A: random split baseline |

| `src/04\_honest\_eval.py` | Experiment B: train Mon-Wed, test Thu-Fri |

| `src/05\_ordered\_split.py` | Experiment C: ordered split within files |

| `src/06\_compare.py` | Random forest vs XGBoost on B and C |

| `src/07\_shap.py` | SHAP feature importance |

| `src/08\_ablation.py` | Feature ablation |



Requires Python 3.10+ and about 8 GB of RAM (developed on 16 GB).



\## Reference



Sharafaldin, I., Lashkari, A. H., Ghorbani, A. A. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." ICISSP 2018.

