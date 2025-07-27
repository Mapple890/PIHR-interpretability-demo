
# PIHR Interpretability Demonstration

This repository demonstrates the interpretability mechanisms of the Physics-Informed Hierarchical Reasoning (PIHR) framework for thickener fault diagnosis.

## Overview

This demo shows how PIHR provides interpretable diagnoses through:
1. Temporal state evolution (p1 → p3a → p2)
2. Feature decomposition (trend vs. volatility)
3. Attention weight visualization
4. Clear decision paths

## Quick Start

```bash
# Clone the repository
git clone https://github.com/[USTB001]/PIHR-interpretability-demo.git
cd PIHR-interpretability-demo

# Install requirements
pip install -r requirements.txt

# Run the demo
jupyter notebook synthetic_case_study.ipynb
