# Early Sepsis Prediction from ICU Data with an ANN

An end-to-end tabular deep learning project: raw hourly ICU records → cleaning → leak-free patient-level split → neural network → evaluation with the official PhysioNet/CinC 2019 clinical utility score.

Sepsis is a life-threatening response to infection, and every hour of delayed treatment raises mortality. The goal is to flag patients **before** clinicians recognize sepsis, using only routine vitals, labs, and demographics.

## Dataset

[PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) (open access).

- 40,336 ICU patients from two hospital systems (A and B), about 1.42 million hourly rows
- One row per patient per hour: 8 vitals, 26 labs, 6 demographic/administrative columns
- `SepsisLabel` is 1 from 6 hours before Sepsis-3 onset, and stays 1 until the end of the stay
- Severe imbalance: **1.85% of training rows are positive**
- Labs are 83-99% missing at row level (measured roughly daily), vitals 10-33% missing (measured roughly hourly)

## Approach

### 1. Cleaning
- `FiO2` outside [0, 1] set to NaN (values up to 4000 and -50 were present)
- `BaseExcess` outside ±40 set to NaN (max was 100)
- Cleaned with fixed clinical limits, not data-driven quantiles, so nothing leaks from the whole dataset

### 2. Missing values (the main preprocessing decision)
1. **"Measured" flags**: one binary column per vital/lab, created *before* filling, so the model can tell a fresh reading from a carried-forward one (ordering a lab is itself a clinical signal)
2. **Forward-fill within each patient** (`groupby("patient_id").ffill()`), which only copies a patient's own past values and cannot leak across patients
3. **Median imputation + standardization** for what remains (hours before a first measurement, never-measured labs), **fitted on the training set only**

Final input: 72 features (34 vitals/labs + 34 measured flags + 4 static columns).

### 3. Leak-free split
- Split **by patient**, never by row: 70% train / 15% validation / 15% test
- Overlap between the three sets verified to be 0 patients
- Thresholds are tuned on validation; the test set is scored once

### 4. Models
| | Base | Improved |
|---|---|---|
| Layers | 512-256-128-64-32 | 256-128-64 |
| Regularization | none | Dropout 0.4 after each hidden layer |
| Callbacks | EarlyStopping | EarlyStopping + ReduceLROnPlateau |
| Loss / optimizer | Binary cross-entropy / Adam (1e-3) | same |
| Class weight (positive) | neg/pos ≈ 53 | neg/pos ≈ 53 |
| Batch size | 2048 | 2048 |

### 5. Evaluation
- **AUROC and AUPRC** (threshold-free). AUPRC is compared with the positive rate (0.017), since accuracy is meaningless at this imbalance
- **Normalized utility score** from the [official challenge code](https://github.com/physionetchallenges): it rewards alarms 6-12 hours before onset and penalizes late alarms, missed cases, and false alarms. 1 is a perfect score and 0 is a model that never raises an alarm
- Decision threshold chosen on validation to maximize utility

## Results

Test set: 212,236 hourly rows, positive rate 1.69%.

| Model | Test AUROC | Test AUPRC | Val utility | **Test utility** | Threshold |
|---|---|---|---|---|---|
| Base | 0.791 | 0.083 | 0.332 | 0.311 | 0.60 |
| Improved (dropout + LR schedule) | 0.808 | 0.099 | 0.362 | **0.354** | 0.55 |

For context, the top teams in the original challenge scored about 0.43 on hospitals A and B (their official ranking used a third, hidden hospital, where scores were much lower).

The improved model at its F1-optimal threshold (0.69) catches about 34% of sepsis rows with 11% precision. At this prevalence, precision is hard to raise without losing early detections, which is exactly the trade-off the utility score measures.

### Notes on reliability
- No random seed was fixed, and repeated runs of these same setups shifted test utility by about ±0.03 (base 0.31-0.34, improved 0.35-0.36). The improved model was ahead in every run, but the size of the gap is not precise.
- **An early version of this project had inflated results** (utility of 0.64) because a bug in the second split reused indices from the temporary subset on the full data, so validation and test patients overlapped with training. It was caught by checking patient overlap between splits, fixed, and all numbers above come from the corrected pipeline.

## Limitations and next steps
- **Single-hour input.** The model sees one hour at a time. Rolling means, trends over the last 6 hours, and "hours since last measured" for each lab are the obvious next features.
- **Row-level labels.** `SepsisLabel` stays 1 after onset, so rows after onset are easy positives. Metrics focused on the pre-onset window would show early detection more directly.
- **No tabular baseline yet.** Gradient boosting (XGBoost/LightGBM) often matches or beats ANNs on tabular data and should be compared on the same split and utility score.
- **Mixed hospitals.** Training on hospital A and testing on B would measure cross-hospital generalization, the main difficulty reported by the challenge.
- **Not a clinical tool.** This is a learning and portfolio project, not validated for patient care.

## Repository

```
.
├── Sepsis.ipynb                 # full pipeline: cleaning, split, models, evaluation
├── evaluate_sepsis_score.py     # official challenge scoring code
├── data/                        # training_setA/, training_setB/ (not included, download from PhysioNet)
└── README.md
```

## How to run
1. Download the training data from PhysioNet and place `training_setA/` and `training_setB/` in `data/`
2. `pip install pandas numpy scikit-learn keras tensorflow pyarrow matplotlib`
3. Merge the `.psv` files into one table (one row per patient-hour, with a `patient_id` column) and save it as Parquet
4. Run `Sepsis.ipynb` from top to bottom

## References
- Reyna MA, Josef CS, Jeter R, et al. *Early Prediction of Sepsis from Clinical Data: the PhysioNet/Computing in Cardiology Challenge 2019.* Critical Care Medicine, 2020.
- Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 2000.
