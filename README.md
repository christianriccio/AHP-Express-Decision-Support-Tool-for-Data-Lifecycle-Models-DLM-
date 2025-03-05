# AHP-Express Decision Support Tool for Data Lifecycle Models comparison

This repository presents a **Streamlit-based application** implementing the **AHP-Express** method for **multi-interview decision support** in **Data Lifecycle Models (DLMs)** selection and evaluation. The simplified AHP approach, proposed by [^1], reduces cognitive bias and the number of required pairwise comparisons by evaluating each element **against a single reference item** instead of performing a full pairwise comparison.

This implementation is specifically designed for **DLM assessment** but is fully adaptable to any other **multi-criteria decision-making (MCDM)** context, such as:
- Comparing multiple alternatives under multiple factors and subfactors,
- Aggregating expert opinions via **multi-interviews**, and
- Simplifying the traditional **AHP** methodology by avoiding the need to construct and process a full $n \times n$ comparison matrix.

---

## 📌 1. Theoretical Background

### 1.1 "Traditional" AHP
The **Analytic Hierarchy Process (AHP)**, introduced by [^2], requires full pairwise comparisons of all elements, resulting in:
- $\frac{n(n-1)}{2}$ judgments for *n* items,
- Increased cognitive effort and risk of inconsistency for larger *n*.

### 1.2 AHP-Express
The **AHP-Express** method reduces these comparisons to **n-1**, by:
1. **Selecting a reference item** for each group (criteria, subcriteria, or alternatives),
2. Comparing each other item *i* to the reference item *R*:
   - If *i* is 3x more important than *R*, then $r_{i} = 3$
   - If *R* is 2x more important than *i*, then $r_{i} = 1/2$
3. Assigning an unnormalized weight $w_{R} = 1$ to the reference,
4. Computing unnormalized weights as $w_{i} = r_{i}$,
5. Normalizing all weights so that their sum is 1.

This leads to fewer comparisons, eliminating the need for **consistency ratio checks**, as inconsistency mainly arises from comparing **low-priority** alternatives.

### 1.3 Multi-Interview Aggregation
AHP-Express can integrate multiple expert opinions using the **geometric mean**, as recommended by Saaty. This ensures robustness by capturing diverse perspectives in a single priority model.

---

## 🔍 2. Code Overview
The code is structured as an interactive **Streamlit application**.

### 2.1 Structure of the Code
+ **Functions:**
  - `normalize_vector()`: Normalizes weights so they sum to 1.
  - `calculate_ahp_express_prior()`: Computes AHP-Express priority scores.
  - `media_geometrica_custom()`: Computes geometric mean for multi-interview aggregation.
  - `create_bar_chart()`, `create_radar_chart()`: Generate visualizations (bar charts, radar charts).
  - `sensitivity_anal()`: Conducts a sensitivity analysis of results.

+ **Main App Flow:**
  1. **Step 1**: Load an **Excel/CSV** file containing **DLMs** and evaluation factors.
  2. **Step 2**: Configure **macro-categories** (e.g., *A = High Data Volume*, *B = Low Data Volume*).
  3. **Step 3**: Conduct **pairwise comparisons** for subfactors using **AHP-Express**.
  4. **Step 4**: Aggregate multi-expert evaluations using **geometric mean**.
  5. **Step 5**: Compute final scores by **weighting alternatives across all factors**.
  6. **Step 6**: Visualize the results via **bar charts, radar plots**, and **sensitivity analysis**.

### 2.2 Key Functions and Code Snippets

#### Pairwise Comparison Implementation
```python

```

#### Final Score Computation
```python

```

## 📊 3. Results Interpretation

Each DLM alternative is assigned a final score based on:

- Factor weights,
- Sub- factor weights,
- Alternative performance on sub-factors.
- An higher score typically indicates a preferred alternative.
