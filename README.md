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
5. Normalizing all weights so that their sum is 1:
   $w_{i}' = \frac{w_{i}}{\sum_{k}w_{k}}$

This leads to fewer comparisons, eliminating the need for **consistency ratio checks**, as inconsistency mainly arises from comparing **low-priority** alternatives.

### 1.3 Multi-Interview Aggregation
AHP-Express can integrate multiple expert opinions using the geometric mean, as recommended by Saaty (1987). For ( n ) experts providing values ( v_1, v_2, …, v_n ), the geometric mean is:

$\text{Geometric Mean} = \left( \prod_{i=1}^{n} v_i \right)^{1/n}$

When weights are provided for experts, a weighted geometric mean is used:

$\text{Weighted Geometric Mean} = \left( \prod_{i=1}^n v_{i}^{w_i} \right)^{1 / \sum w_{i}}$

This ensures robustness by capturing diverse perspectives in a single priority model.

---

## 🔍 2. Code Overview
The code is structured as an interactive **Streamlit application**.

### 2.1 Structure of the Code
The application is divided into:
- **Main Function (main())**: Orchestrates the Streamlit interface, guiding users through data input, configuration, comparisons, and result visualization.
- **Helper Functions**: Perform specific tasks like data loading, priority calculations, and visualizations.

### Functions
1. `normalize_vector(v)`: Normalizes a vector so its elements sum to 1, ensuring AHP priorities are consistent.
```python
def normalize_vector(v):
    """function to normalize a vector (sum to 1)"""
    s = sum(v)
    if s == 0:
        return [0] * len(v)
    return [x / s for x in v]
```

+ **Usage**: Applied after computing unnormalized priorities to ensure they sum to 1.
  
2. `calculate_ahp_express_prior(base_index, values)`: Computes AHP-Express priority scores by comparing all elements against a reference (base) item.
```python
def calculate_ahp_express_prior(base_index, values):
    """
    This function is responsible for calculating the priority vector according AHP-express
    Formula: pr_j = (1/a(base,j)) / Σ_k (1/a(base,k))
    """
    reciprocals = [1.0 / v for v in values]
    denom = sum(reciprocals)
    priorities = [(1.0 / v) / denom for v in values]
    return priorities
```

+ **Usage**: Takes comparison values (e.g., Saaty’s scale: 1, 3, 9) and returns normalized priorities.

3. `media_geometrica_custom(valori, pesi=None)`: Computes the geometric mean for multi-interview aggregation, optionally weighted.
```python
def media_geometrica_custom(valori, pesi=None):
    """
    This function calculates the geometric mean:
      - if no weights are provided, use standard geometric mean
      - if weights are provided, use weighted geometric mean
    """
    valori_filtrati = [v for v in valori if v > 0]
    if len(valori_filtrati) == 0:
        return 1.0
    if pesi is None:
        prodotto = 1.0
        for v in valori:
            if v > 0:
                prodotto *= v
        return prodotto ** (1.0 / len(valori_filtrati))
    else:
        if len(pesi) != len(valori):
            st.error("Weights lengths does not match values.")
            return 1.0
        tot = sum(pesi)
        norm_pesi = [w / tot if tot > 0 else 1.0 / len(pesi) for w in pesi]
        prodotto = 1.0
        for v, w in zip(valori, norm_pesi):
            if v > 0:
                prodotto *= v ** w
        return prodotto
```

+ **Usage**: Aggregates comparison values from multiple interviews, ensuring robust prioritization.

4. `create_bar_chart(labels, values, title="Bar Chart")` and `create_radar_chart(df, title="Radar Chart")`: Generate visualizations for DLM scores and sub-factor comparisons.

5. `sensitivity_anal(df_dlm, sottofattori, priA_norm, priB_norm)`: Conducts a sensitivity analysis by varying macro-category weights.
```python
def sensitivity_anal(df_dlm, sottofattori, priA_norm, priB_norm):
    """
    This function is responsible for calculating the sensitivity analysis by varying the weight of cat. A from 0 to 1.
    """
    st.header("Sensitivity Analysis")
    pA_values = np.linspace(0, 1, 21)
    sensitivity_scores = {dlm: [] for dlm in df_dlm[df_dlm.columns[0]].tolist()}
    for pA in pA_values:
        pB = 1 - pA
        combined = [priA_norm[i] * pA + priB_norm[i] * pB for i in range(len(sottofattori))]
        combined_norm = normalize_vector(combined)
        for index, row in df_dlm.iterrows():
            score = sum(float(row[sf]) * combined_norm[i] for i, sf in enumerate(sottofattori))
            dlm_name = row[df_dlm.columns[0]]
            sensitivity_scores[dlm_name].append(score)
```
 ### Main App Flow:
  1. **Step 1**: Load an **Excel/CSV** file containing **DLMs** and evaluation factors.
  2. **Step 2**: Configure **macro-categories** (e.g., *A = New and Big Data*, *B = Old data*).
  3. **Step 3**: Conduct **pairwise comparisons** for subfactors using **AHP-Express**.
  4. **Step 4**: Aggregate multi-expert evaluations using **geometric mean**.
  5. **Step 5**: Compute final scores by **weighting alternatives across all factors**.
  6. **Step 6**: Visualize the results via **bar charts, radar plots**, and **sensitivity analysis**.

### Input Data Requirements
To use the tool, users need to upload a dataset in **Excel (.xlsx)** or **CSV (.csv)** format containing the alternatives to be evaluated and their respective sub-factor scores. The structure of the file should follow these points:

#### Required Data Format
- **First Column:** The name or identifier of each alternative (e.g., different Data Lifecycle Models - DLMs)
- **Subsequent Columns:** Scores for each sub-factor under evaluation

#### Example Structure of Input File

| ALternative Name |Factor 1       | Factor 2       |
| ---------------- | ------------- | -------------  |
| DLM A            | Score 1       | Score 3        |
| DLM B            | Score 2       | Score 4        |

#### Additional Considerations

- Numeric values only in sub-factor columns: The tool requires numerical scores (e.g., performance metrics, evaluations, or expert-assigned weights).
- Consistent scaling: All sub-factor values should be expressed in the same scale to ensure meaningful comparisons.
- No missing values: Missing data may impact the calculations. Ensure all alternatives have scores for every sub-factor.
- Customizable categories: The tool allows defining macro-categories (e.g., New Data, Old Data) to group sub-factors.


#### Pairwise Comparison Implementation

The AHP-Express method is implemented in `calculate_ahp_express_prior()`, reducing comparisons to ( n-1 ). For example, with sub-factors [S1, S2, S3] and S1 as the reference, users provide comparisons like S1 vs. S2 = 3 and S1 vs. S3 = 9, which are processed to derive priorities.

#### Final Score Computation
Final DLM scores are computed in `main()` by multiplying DLM sub-factor values by the combined priority vector:

```python
data_scores = []
for _, row in df_dlm.iterrows():
    score = sum(float(row[sf]) * final_subfactors[i] for i, sf in enumerate(sottofattori))
    data_scores.append(score)
df_dlm['Final_score'] = data_scores
```
This weighted sum ensures each DLM performance is evaluated against all sub-factors, weighted by their final priorities.
## 📊 3. Results Interpretation

Each DLM alternative is assigned a final score based on:
- **Factor weight**s: Macro-category weights (0.5 default for both cat. A and for cat. B).
- **Sub-factor weights**: Priorities derived from AHP-Express comparisons.
- **Alternative performance on sub-factors**: DLM scores for each sub-factor in the input file.

A higher score indicates a preferred alternative. The tool provides:
- **Bar Chart**: Displays final DLM scores for ranking.
- **Radar Chart**: Compares DLMs across sub-factors, normalized for clarity.
- **Sensitivity Analysis**: Shows how scores vary with macro-category weights, assessing robustness.



## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Required libraries: streamlit, pandas, numpy, matplotlib, seaborn

### Installation
1. **Clone this repository via:**
    ```console
    git clone
    ```
    and move inside the cloned directory 

2. **Install dependencies:**
 
```console
pip install -r requirements.txt
```

3. **Run the application:**
 ```console
streamlit run ahp3.py
```
ore you can use the tool as a streamlit web application at the following [link](https://ahp-express-dlm.streamlit.app/). 

### Usage
1. Upload an Excel or CSV file with DLMs and sub-factor scores (first column = DLM name, subsequent columns = sub-factors).
2. Configure macro-category weights and the number of interviews.
3. Perform AHP-Express comparisons for each category and interview.
4. Review the final rankings, visualizations, and sensitivity analysis.

### 📝 Acknowledgments
> [!NOTE]
> Feel free to use or modify the code as long as you mention this repository and its contributors. For further details, contact the primary author at christian.riccio@unicampania.it.

## References 
This implementation builds on the theorethical foundations of:

[^1]: LEAL, José Eugenio. AHP-express: A simplified version of the analytical hierarchy process method. MethodsX, 2020, 7: 100748.
[^2]: SAATY, Roseanna W. The analytic hierarchy process—what it is and how it is used. Mathematical modelling, 1987, 9.3-5: 161-176.
