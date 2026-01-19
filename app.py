import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Konfiguracja strony
st.set_page_config(page_title="Kalkulator Teorii Gier", page_icon="🎲")

st.title("🎲 Kalkulator Teorii Gier")
st.markdown("---")

menu = st.sidebar.selectbox("Wybierz moduł:", ["Gry z Naturą", "Gry 2-osobowe (Suma Zero)"])

def solve_mixed(matrix):
    m, n = matrix.shape
    offset = abs(np.min(matrix)) + 1
    M = matrix + offset
    c = [-1] * m
    res = linprog(c, A_ub=-M.T, b_ub=[1] * n, bounds=(0, None), method='highs')
    if res.success:
        p = res.x / np.sum(res.x)
        v = 1/np.sum(res.x) - offset
        return v, p
    return None, None

if menu == "Gry z Naturą":
    st.header("🌿 Analiza Decyzyjna: Gry z Naturą")
    col_input, col_params = st.columns([2, 1])
    
    with col_input:
        data_input = st.text_area("Wpisz macierz (spacje = kolumny, nowe linie = wiersze):", 
                                 "85 75 95\n85 90 75.5\n85 65 98")
    with col_params:
        alfa = st.slider("Współczynnik Hurwicza (α):", 0.0, 1.0, 0.6)

    if data_input:
        matrix = np.array([list(map(float, row.split())) for row in data_input.split('\n') if row.strip()])
        st.subheader("Macierz wypłat")
        st.dataframe(matrix)

        # Obliczenia
        mins = np.min(matrix, axis=1)
        maxs = np.max(matrix, axis=1)
        hurwicz = alfa * mins + (1 - alfa) * maxs
        bayes = np.mean(matrix, axis=1)
        
        # Savage
        regret_matrix = np.max(matrix, axis=0) - matrix
        savage = np.max(regret_matrix, axis=1)

        res_df = pd.DataFrame({
            'Wald (Maximin)': mins,
            'Hurwicz': hurwicz,
            'Bayes (Laplace)': bayes,
            'Savage (Minimax Żalu)': savage
        }, index=[f"Strategia {i+1}" for i in range(len(matrix))])
        
        st.subheader("📊 Wyniki porównawcze")
        st.table(res_df)
        st.success(f"Najlepszy wybór wg Walda: Strategia {np.argmax(mins)+1}")

elif menu == "Gry 2-osobowe (Suma Zero)":
    st.header("⚔️ Gry Dwuosobowe (A vs B)")
    matrix_text = st.text_area("Macierz dla Gracza A:", "-4 5 6\n2 3 3\n-5 4 5")
    
    if matrix_text:
        A = np.array([list(map(float, row.split())) for row in matrix_text.split('\n') if row.strip()])
        va = np.max(np.min(A, axis=1))
        vb = np.min(np.max(A, axis=0))
        
        st.write(f"**Dolna wartość gry ($v_a$):** {va} | **Górna wartość gry ($v_b$):** {vb}")
        
        if va == vb:
            st.success(f"Punkt siodłowy! Wartość gry = {va}")
        else:
            st.warning("Brak punktu siodłowego. Obliczanie strategii mieszanych...")
            v, p = solve_mixed(A)
            _, q = solve_mixed(-A.T)
            
            st.metric("Wartość gry (v)", round(v, 4))
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Gracz A (Wiersze):**")
                for i, prob in enumerate(p):
                    st.write(f"Z{i+1}: {prob*100:.2f}%")
                    st.progress(float(prob))
            with c2:
                st.write("**Gracz B (Kolumny):**")
                for i, prob in enumerate(q):
                    st.write(f"S{i+1}: {prob*100:.2f}%")
                    st.progress(float(prob))
