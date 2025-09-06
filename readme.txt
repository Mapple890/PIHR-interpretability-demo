# PIHR Interpretability and Generalization Demonstration

This repository provides a minimal, self-contained, and runnable demonstration of the core concepts presented in our paper: **"Physics-Informed Hierarchical Reasoning (PIHR): A Structured Framework for Generalizable Fault Diagnosis in Industrial Thickeners."**

## Overview

The single Python script, `demonstration.py`, performs a complete, miniature research study that validates the central scientific claim of our paper: **zero-shot generalization.**

The script will:
1.  **Programmatically generate** a high-quality synthetic dataset simulating a classic thickener compaction fault, evolving from a stable state (`p1`) to a gradual rise (`p3a`) and finally to a critical accelerated rise (`p2`).
2.  **Implement** a complete, trainable version of our PIHR architecture in PyTorch.
3.  **Train the model** using a curriculum where it is **exclusively exposed to the "seen" foundational states** (`p1` and `p3a`).
4.  **Evaluate the trained model** on a test set that includes the **"unseen" complex fault state (`p2`)**, for which the model has never seen a label.
5.  **Print a final classification report and confusion matrix** to quantitatively demonstrate the model's high accuracy on this zero-shot diagnostic task.

## Quick Start

To run this demonstration, you will need Python 3.x and Git installed.

**1. Clone the repository:**
```bash
git clone https://github.com/Mapple890/PIHR-interpretability-demo.git
cd PIHR-interpretability-demo
```

**2. Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**3. Install the required libraries:**
```bash
pip install -r requirements.txt
```

**4. Run the demonstration:**
```bash
python demonstration.py
```

## Expected Output

The script will print the training progress for each epoch and will conclude with a final report summarizing the model's performance. The key result to note is the high recall and precision for the `p2_Accelerated` class, confirming the model's ability to successfully generalize.

*An example of the final output:*
```
============================================================
FINAL RESULTS
============================================================

Classification Report:
                precision    recall  f1-score   support

     p1_Stable       0.71      1.00      0.83       100
   p3a_Gradual       1.00      0.59      0.74       100
p2_Accelerated       1.00      1.00      1.00       100

      accuracy                           0.86       300
     macro avg       0.90      0.86      0.86       300
  weighted avg       0.90      0.86      0.86       300


Confusion Matrix:
[[100   0   0]
 [ 41  59   0]
 [  0   0 100]]

============================================================
KEY INSIGHT:
SUCCESS! PIHR achieved 100.0% accuracy on the UNSEEN p2 class!
============================================================
```
```

