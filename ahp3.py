import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import io
import csv
import requests
import seaborn as sns


testo_introduzione = r"""
# AHP-express Decision Tool for DLM comparison with Multi-Interviews Support

This **tool** is designed to support anyone who needs to **select**, **evaluate**, 
or **prioritize** *Data Lifecycle Models* (DLM) without necessarily having 
a specialized background in decision methodologies or AHP. 
Here’s a brief overview of how it works:

1. **AHP and AHP-express**  
   - The Analytic Hierarchy Process (**AHP**) is a multi-criteria decision-making 
     method that involves pairwise comparisons of the elements of a problem 
     (criteria, sub-factors, alternatives) to calculate priorities.  
   - **AHP-express** is a **simplified version** of traditional AHP (Saaty) 
     that drastically reduces the number of required comparisons.  
     Instead of comparing *all against all*, it selects a **"base" element** 
     (dominant) and evaluates only *"base vs. others"*. 

2. **Macro-categories (A and B)**  
   - Often, before analyzing sub-factors in detail, there are *macro-categories* 
     (in our case, “A” and “B”), such as the question:
     “Are we managing a project with a large amount of data (A) or not (B)?”
   - Each macro-category can be assigned a **weight** (e.g., 0.5 for A, 0.5 for B) 
     that defines its overall importance.

3. **Sub-factors**  
   - For each macro-category, **sub-factors** (more specific criteria) 
     that characterize the DLMs are defined.  
   - These are also compared using the AHP-express logic, 
     selecting a base sub-factor and comparing it with the others.

4. **Multi-interviews and geometric mean**  
   - If multiple people/experts contribute to the evaluation (e.g., different stakeholders), 
     *separate interviews* are conducted, resulting in multiple sets of comparisons.  
   - To combine all evaluations into a single final result, we use the 
     **geometric mean** of each pairwise comparison (as recommended by Saaty for AHP).

5. **DLM and final scores**  
   - Once the priorities of the sub-factors are calculated, 
     and the macro-category weights are combined, 
     we can assign each DLM a **total score** 
     by computing the weighted sum of the DLM values for each sub-factor 
     (e.g., [DLM value for SF1]*[priority of SF1] + …).  
   - The tool displays the **final ranking** and provides summary charts 
     (a *Bar Chart* and a *Radar Chart*) for better result interpretation.

**Getting Started**:  
- Upload your *Excel/CSV* file containing the DLMs and their sub-factors or;
- Download and then use a pre-configured CSV file at the following link:  
- Set the number of interviews, macro-category weights, and for each interview, 
  enter the comparisons “base vs. others”.  
- Press the button to calculate the final results.
"""


DEFAULT_DLM_CSV = """Starting;Assessment;Computation;Administration;Security;End-of-life
USGS;8;10;5;3;0;5
DataONE;4;7;10;2;0;0
IBM;5;0;2;5;9;4
Hindawi;5;10;8;6;7;5
DCC;7;7;8;10;6;4
CRUD;5;7;4;7;10;8
CIGREF;8;7;9;6;5;0
DDI;8;7;5;3;0;6
PII;5;4;4;6;10;4
EDLM;4;0;6;7;2;9
"""


def carica_dlm_da_file(uploaded_file):
    """
    This function is responsible for loading the DLM data from an excel/csv file separated by different separators.
    """
    filename = uploaded_file.name.lower()
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            return df
        except Exception as e:
            st.error(f"error reading Excel: {e}")
            return None
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        return df
    except Exception as e:
        st.error(f"Attempt 1 (autodetect) failed: {e}")
        try:
            uploaded_file.seek(0)
            rawdata = uploaded_file.read(2048).decode('utf-8', errors='replace')
            uploaded_file.seek(0)
            dialect = csv.Sniffer().sniff(rawdata, delimiters=[',', ';', '\t', '|'])
            sep = dialect.delimiter
            df = pd.read_csv(uploaded_file, sep=sep)
            return df
        except Exception as e2:
            st.error(f"Attempt 2 (csv.Sniffer) failed: {e2}")
            return None

def normalize_vector(v):
    """function to normalize a vector (sum to 1)"""
    s = sum(v)
    if s == 0:
        return [0] * len(v)
    return [x / s for x in v]

def calculate_ahp_express_prior(base_index, values):
    """
    This function is responsible for calculating the priority vector according AHP-express
    Formula: pr_j = (1/a(base,j)) / Σ_k (1/a(base,k))
    """
    reciprocals = [1.0 / v for v in values]
    denom = sum(reciprocals)
    priorities = [(1.0 / v) / denom for v in values]
    return priorities

def media_geometrica_custom(valori, pesi=None):
    """
    This function calculates the geometric mean:
      - if no weights are provided, use standard geometric mean
      - if weigths are provided, use weighted geometric mean

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

def create_bar_chart(labels, values, title="Bar Chart"):
    """bar-chart plot"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_xlabel("DLM")
    st.pyplot(fig)

def create_radar_chart(df, title="Radar Chart"):
    """Create a radar plot of the sub-factors to compare different DLMs"""

    categories = list(df.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for index, row in df.iterrows():
        values = row.values.tolist()
        values += values[:1]
        ax.plot(angles, values, label=index)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)

def sensitivity_anal(df_dlm, sottofattori, priA_norm, priB_norm):
    """
    This function is responsible for calculating the sensitivity analysis by varying the weight of cat. A from 0 to 1.
    Shows:
    - a line graph showing the variation of the scores by varion of the weight of cat.A
    - a table showing statistics (min,max, delta) for each DLM
    - dynamic comments explaining the meaning of each value
    """
    st.header("Sensitivity Analysis")

    pA_values = np.linspace(0, 1, 21)
    sensitivity_scores = {dlm: [] for dlm in df_dlm[df_dlm.columns[0]].tolist()}

    for pA in pA_values:
        pB = 1 - pA
        # calculate combined vector for subfactors
        combined = [priA_norm[i] * pA + priB_norm[i] * pB for i in range(len(sottofattori))]
        combined_norm = normalize_vector(combined)
        # calculate scores for each DLM
        for index, row in df_dlm.iterrows():
            score = sum(float(row[sf]) * combined_norm[i] for i, sf in enumerate(sottofattori))
            dlm_name = row[df_dlm.columns[0]]
            sensitivity_scores[dlm_name].append(score)

    df_sens = pd.DataFrame(sensitivity_scores, index=pA_values)
    df_sens.index.name = "Cat.A weight"


    st.subheader("Scores variation of DLM")
    fig, ax = plt.subplots(figsize=(10, 6))
    for dlm in df_sens.columns:
        ax.plot(df_sens.index, df_sens[dlm], marker='o', label=dlm)
    ax.set_xlabel("Weight of category A")
    ax.set_ylabel("DLM score")
    ax.set_title("Scores sensitivity of DLMs by weight of cat. A")
    ax.legend()
    st.pyplot(fig)


    stats = {dlm: {"Min": np.min(scores), "Max": np.max(scores), "Delta": np.max(scores) - np.min(scores)}
             for dlm, scores in sensitivity_scores.items()}
    df_stats = pd.DataFrame(stats).T

    st.subheader("Sensitivity statistics")
    st.dataframe(df_stats)


    st.subheader("Comments about sensitivity")
    for dlm, row in df_stats.iterrows():
        comment = (f"For the DLM **{dlm}**: the minimum score obtained is **{row['Min']:.2f}**, "
                   f"the maximum score is **{row['Max']:.2f}**, and the difference (Delta) is **{row['Delta']:.2f}**. "
                   "A high Delta indicates that the model is highly sensitive to variations in the weight "
                   "of Category A, while a low Delta suggests greater stability in the evaluation.")

        st.write(comment)

    return df_sens

def saaty_scale_description():
    """Returns a dictionary of Saaty's scale"""
    return {
        1: "EQUAL importance",
        2: "Equal importance with slight preference",
        3: "MODERATE importance",
        4: "Moderate importance with strong preference",
        5: "STRONG importance",
        6: "Strong importance with very strong preference",
        7: "VERY STRONG importance",
        8: "Very strong importance with extreme preference",
        9: "EXTREME importance"
    }


def main():
    st.markdown(testo_introduzione, unsafe_allow_html=True)
    st.title("AHP-Express Tool to compare DLMs")

   
    st.header("1. Load DLMs file or use defaults")
    mode = st.radio("Choose the Input", 
                    ("Upload your own CSV/Excel", "Use default pre-configured DLMs"))
    if mode == "Upload your own CSV/Excel":
        uploaded_file = st.file_uploader("Upload Excel/CSV file with DLMs", type=['csv','xlsx','xls'])
        if not uploaded_file:
            st.info("Upload a file to proceed.")
            return
        df_dlm = carica_dlm_da_file(uploaded_file)
        if df_dlm is None or df_dlm.empty:
            st.error("File not valid or empty")
            return
    else:
        raw_url = "https://raw.githubusercontent.com/christianriccio/AHP-Express-Decision-Support-Tool-for-Data-Lifecycle-Models-DLM-/main/dlms.csv"
        st.markdown(f"[Download the pre-configured file]({raw_url})", unsafe_allow_html=True)
        df_dlm = pd.read_csv(io.StringIO(DEFAULT_DLM_CSV), sep=';')
        st.success("Use of the 10 pre-confgirued DLMs.")

    st.subheader("Data preview")
    st.dataframe(df_dlm.head(10))


    dlm_names = df_dlm[df_dlm.columns[0]].tolist()
    sottofattori = list(df_dlm.columns[1:])


    st.header("2. Configure Interviews")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Macro-Categories")
        usa_pesi_macro = st.checkbox("Enable macro-category weights", value=True)
        if usa_pesi_macro:
            peso_A = st.slider("Weight for Category A (Large Data)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            peso_B = 1 - peso_A
            st.write(f"Weight for Category B (Old Data): {peso_B:.2f}")
        else:
            peso_A, peso_B = 0.5, 0.5
    with col2:
        st.subheader("interviews")
        num_interviste = st.number_input("Number of Interviews", min_value=1, max_value=10, value=1)


    st.header("3. AHP-Express Comparisons (Saaty's Scale)")

    st.subheader("Category A - Many Data")
    base_sottofattore_A = st.selectbox("Reference sub-factor", sottofattori, key="baseA")
    comparisons_A = {sf: [] for sf in sottofattori}
    interview_weights_A = []
    for i in range(num_interviste):
        st.markdown(f"**Interview A #{i + 1}**")
        peso_intervista = st.slider(f"Weight of interview A #{i + 1}", min_value=0.0, max_value=2.0, value=1.0, step=0.1,
                                    key=f"weightA_{i}")
        interview_weights_A.append(peso_intervista)
        for sf in sottofattori:
            if sf == base_sottofattore_A:
                comparisons_A[sf].append(1.0)
            else:
                val = st.select_slider(f"Relative importance {base_sottofattore_A} vs {sf}",
                                       options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                       value=1,
                                       key=f"compA_{i}_{sf}",
                                       format_func=lambda x: f"{x}: {scala_saaty[x]}")
                comparisons_A[sf].append(val)

    st.subheader("Category B - Old Data")
    base_sottofattore_B = st.selectbox("Reference sub-factor", sottofattori, key="baseB")
    comparisons_B = {sf: [] for sf in sottofattori}
    interview_weights_B = []
    for i in range(num_interviste):
        st.markdown(f"**Interview B #{i + 1}**")
        peso_intervista = st.slider(f"Weight of interview B #{i + 1}", min_value=0.0, max_value=2.0, value=1.0, step=0.1,
                                    key=f"weightB_{i}")
        interview_weights_B.append(peso_intervista)
        for sf in sottofattori:
            if sf == base_sottofattore_B:
                comparisons_B[sf].append(1.0)
            else:
                val = st.select_slider(f"Relative importance {base_sottofattore_B} vs {sf}",
                                       options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                       value=1,
                                       key=f"compB_{i}_{sf}",
                                       format_func=lambda x: f"{x}: {scala_saaty[x]}")
                comparisons_B[sf].append(val)


    if st.button("Calculate Priorities and Analysis"):
        # Cat. A priority calculation
        base_index_A = sottofattori.index(base_sottofattore_A)
        final_ratios_A = [media_geometrica_custom(comparisons_A[sf], interview_weights_A)
                          for sf in sottofattori]
        priA = calculate_ahp_express_prior(base_index_A, final_ratios_A)
        priA_norm = normalize_vector(priA)
        # Cat. B priority calculation
        base_index_B = sottofattori.index(base_sottofattore_B)
        final_ratios_B = [media_geometrica_custom(comparisons_B[sf], interview_weights_B)
                          for sf in sottofattori]
        priB = calculate_ahp_express_prior(base_index_B, final_ratios_B)
        priB_norm = normalize_vector(priB)
        # final priority vector combined
        final_subfactors = [priA_norm[i] * peso_A + priB_norm[i] * peso_B for i in range(len(sottofattori))]
        final_subfactors = normalize_vector(final_subfactors)


        st.header("Priority Vectors")
        df_priA = pd.DataFrame({"Sub-factor": sottofattori, "Priority Category A": priA_norm})
        st.subheader("Category A")
        st.dataframe(df_priA)
        df_priB = pd.DataFrame({"Sub-factor": sottofattori, "Priority Category B": priB_norm})
        st.subheader("Category B")
        st.dataframe(df_priB)
        df_final = pd.DataFrame({"Sub-factor": sottofattori, "Priority Combined": final_subfactors})
        st.subheader("Sub-factors final priority")
        st.dataframe(df_final)


        data_scores = []
        for _, row in df_dlm.iterrows():
            score = sum(float(row[sf]) * final_subfactors[i] for i, sf in enumerate(sottofattori))
            data_scores.append(score)
        df_dlm['Final_score'] = data_scores


        st.header("Results")
        st.subheader("DLMs Rank")
        data_sorted = df_dlm.sort_values(by="Final_score", ascending=False)
        st.dataframe(data_sorted)

        st.subheader("Bar Plot")
        create_bar_chart(data_sorted[df_dlm.columns[0]].tolist(), data_sorted["Final_score"].tolist(),
                       title="Final scores of DLMs")

        st.subheader("Radar Plot")
        df_radar = pd.DataFrame(index=df_dlm[df_dlm.columns[0]], columns=sottofattori)
        for _, row in df_dlm.iterrows():
            nome = row[df_dlm.columns[0]]
            for i, sf in enumerate(sottofattori):
                try:
                    val = float(row[sf])
                except:
                    val = 0.0
                df_radar.loc[nome, sf] = val
        # normalize each sub-factor
        for sf in sottofattori:
            col = df_radar[sf].astype(float)
            max_val = col.max()
            if max_val > 0:
                df_radar[sf] = col / max_val
            else:
                df_radar[sf] = 0.0
        create_radar_chart(df_radar, title="DLM vs Sub-factors")


        sensitivity_anal(df_dlm, sottofattori, priA_norm, priB_norm)

if __name__ == "__main__":
    main()
