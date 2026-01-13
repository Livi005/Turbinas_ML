# Logistic Regression Model Comparison Report

**Generated on:** 2025-11-19 01:08:22

**Number of models compared:** 8

## Executive Summary

**Best Overall Model:** `Scaled Train, Scaled Val` (Score: 0.9610)

### Best Model by Metric
| Metric | Best Model | Value |
|--------|------------|-------|
| Accuracy | `Regular (No Aug, No Scale)` | 0.6946 |
| Precision | `Aug SMOTE & Scaled Train, Scaled Val` | 0.5871 |
| Recall | `Regular (No Aug, No Scale)` | 0.6946 |
| F1 | `Regular (No Aug, No Scale)` | 0.5694 |
| Roc_auc | `Aug SMOTE & Scaled Train, Scaled Val` | 0.5127 |

## Overall Ranking
| Rank | Model Name | Overall Score |
|------|------------|---------------|
| 1 | `Scaled Train, Scaled Val` | 0.9610 |
| 2 | `Scaled Train, Regular Val` | 0.9594 |
| 3 | `Aug SMOTE & Scaled Train, Regular Val` | 0.9594 |
| 4 | `Regular (No Aug, No Scale)` | 0.9593 |
| 5 | `Aug SMOTE & Scaled Train, Scaled Val` | 0.8841 |
| 6 | `Augmented SMOTE Train, Regular Val` | 0.8715 |
| 7 | `Regular Train, Scaled Val` | 0.8211 |
| 8 | `Augmented SMOTE Train, Scaled Val` | 0.8093 |

## Detailed Metrics Comparison
| Model | Accuracy | Precision | Recall | F1 | Roc_auc |
|------|------|------|------|------|------|
| `Regular (No Aug, No Scale)` | 0.6946 | 0.4825 | 0.6946 | 0.5694 | 0.4997 |
| `Regular Train, Scaled Val` | 0.4790 | 0.5559 | 0.4790 | 0.4995 | 0.4625 |
| `Augmented SMOTE Train, Regular Val` | 0.5090 | 0.5743 | 0.5090 | 0.5282 | 0.5056 |
| `Augmented SMOTE Train, Scaled Val` | 0.4731 | 0.5477 | 0.4731 | 0.4938 | 0.4535 |
| `Scaled Train, Regular Val` | 0.6946 | 0.4825 | 0.6946 | 0.5694 | 0.5000 |
| `Scaled Train, Scaled Val` | 0.6946 | 0.4825 | 0.6946 | 0.5694 | 0.5041 |
| `Aug SMOTE & Scaled Train, Regular Val` | 0.6946 | 0.4825 | 0.6946 | 0.5694 | 0.5000 |
| `Aug SMOTE & Scaled Train, Scaled Val` | 0.5150 | 0.5871 | 0.5150 | 0.5341 | 0.5127 |

## Best Model for Each Metric

### Accuracy
- **Best Model:** `Regular (No Aug, No Scale)`
- **Value:** 0.6946

**All Models:**
| Model | Value |
|-------|-------|
| `Regular (No Aug, No Scale)` ⭐ | 0.6946 |
| `Scaled Train, Regular Val`  | 0.6946 |
| `Scaled Train, Scaled Val`  | 0.6946 |
| `Aug SMOTE & Scaled Train, Regular Val`  | 0.6946 |
| `Aug SMOTE & Scaled Train, Scaled Val`  | 0.5150 |
| `Augmented SMOTE Train, Regular Val`  | 0.5090 |
| `Regular Train, Scaled Val`  | 0.4790 |
| `Augmented SMOTE Train, Scaled Val`  | 0.4731 |

### Precision
- **Best Model:** `Aug SMOTE & Scaled Train, Scaled Val`
- **Value:** 0.5871

**All Models:**
| Model | Value |
|-------|-------|
| `Aug SMOTE & Scaled Train, Scaled Val` ⭐ | 0.5871 |
| `Augmented SMOTE Train, Regular Val`  | 0.5743 |
| `Regular Train, Scaled Val`  | 0.5559 |
| `Augmented SMOTE Train, Scaled Val`  | 0.5477 |
| `Regular (No Aug, No Scale)`  | 0.4825 |
| `Scaled Train, Regular Val`  | 0.4825 |
| `Scaled Train, Scaled Val`  | 0.4825 |
| `Aug SMOTE & Scaled Train, Regular Val`  | 0.4825 |

### Recall
- **Best Model:** `Regular (No Aug, No Scale)`
- **Value:** 0.6946

**All Models:**
| Model | Value |
|-------|-------|
| `Regular (No Aug, No Scale)` ⭐ | 0.6946 |
| `Scaled Train, Regular Val`  | 0.6946 |
| `Scaled Train, Scaled Val`  | 0.6946 |
| `Aug SMOTE & Scaled Train, Regular Val`  | 0.6946 |
| `Aug SMOTE & Scaled Train, Scaled Val`  | 0.5150 |
| `Augmented SMOTE Train, Regular Val`  | 0.5090 |
| `Regular Train, Scaled Val`  | 0.4790 |
| `Augmented SMOTE Train, Scaled Val`  | 0.4731 |

### F1
- **Best Model:** `Regular (No Aug, No Scale)`
- **Value:** 0.5694

**All Models:**
| Model | Value |
|-------|-------|
| `Regular (No Aug, No Scale)` ⭐ | 0.5694 |
| `Scaled Train, Regular Val`  | 0.5694 |
| `Scaled Train, Scaled Val`  | 0.5694 |
| `Aug SMOTE & Scaled Train, Regular Val`  | 0.5694 |
| `Aug SMOTE & Scaled Train, Scaled Val`  | 0.5341 |
| `Augmented SMOTE Train, Regular Val`  | 0.5282 |
| `Regular Train, Scaled Val`  | 0.4995 |
| `Augmented SMOTE Train, Scaled Val`  | 0.4938 |

### Roc_auc
- **Best Model:** `Aug SMOTE & Scaled Train, Scaled Val`
- **Value:** 0.5127

**All Models:**
| Model | Value |
|-------|-------|
| `Aug SMOTE & Scaled Train, Scaled Val` ⭐ | 0.5127 |
| `Augmented SMOTE Train, Regular Val`  | 0.5056 |
| `Scaled Train, Scaled Val`  | 0.5041 |
| `Scaled Train, Regular Val`  | 0.5000 |
| `Aug SMOTE & Scaled Train, Regular Val`  | 0.5000 |
| `Regular (No Aug, No Scale)`  | 0.4997 |
| `Regular Train, Scaled Val`  | 0.4625 |
| `Augmented SMOTE Train, Scaled Val`  | 0.4535 |

## Recommendations

### Best Overall Choice
Based on the overall ranking, **`Scaled Train, Scaled Val`** is recommended as the best overall model.

### Metric-Specific Recommendations
- **For Accuracy:** Use `Regular (No Aug, No Scale)` (value: 0.6946)
- **For Precision:** Use `Aug SMOTE & Scaled Train, Scaled Val` (value: 0.5871)
- **For Recall:** Use `Regular (No Aug, No Scale)` (value: 0.6946)
- **For F1:** Use `Regular (No Aug, No Scale)` (value: 0.5694)
- **For Roc_auc:** Use `Aug SMOTE & Scaled Train, Scaled Val` (value: 0.5127)

## Model Configurations

The following models were compared:
1. **`Regular (No Aug, No Scale)`**
2. **`Regular Train, Scaled Val`**
3. **`Augmented SMOTE Train, Regular Val`**
4. **`Augmented SMOTE Train, Scaled Val`**
5. **`Scaled Train, Regular Val`**
6. **`Scaled Train, Scaled Val`**
7. **`Aug SMOTE & Scaled Train, Regular Val`**
8. **`Aug SMOTE & Scaled Train, Scaled Val`**
