from flask import Flask, render_template, request
import numpy as np
import re
import os

# KLUCZOWA ZMIANA: Wskazujemy ścieżkę do templates relatywnie do tego pliku
app = Flask(__name__, template_folder='../templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        try:
            # Pobranie parametru gamma
            gamma_raw = request.form.get('gamma', '0.6').replace(',', '.')
            gamma = float(gamma_raw) if gamma_raw else 0.6
            
            data_dict = {}
            max_row, max_col = -1, -1
            
            # Zbieranie danych z pól formularza
            for key in request.form.keys():
                match = re.match(r'cell_(\d+)_(\d+)', key)
                if match:
                    r, c = map(int, match.groups())
                    val_str = request.form[key].replace(',', '.').strip()
                    val = float(val_str) if val_str else 0.0
                    
                    data_dict[(r, c)] = val
                    max_row = max(max_row, r)
                    max_col = max(max_col, c)
            
            if max_row == -1:
                return render_template('index.html', results=None)

            matrix = np.zeros((max_row + 1, max_col + 1))
            for (r, c), val in data_dict.items():
                matrix[r, c] = val
            
            min_rows = np.min(matrix, axis=1)
            max_rows = np.max(matrix, axis=1)
            
            wald_idx = int(np.argmax(min_rows))
            hurwicz_vals = gamma * min_rows + (1 - gamma) * max_rows
            hurwicz_idx = int(np.argmax(hurwicz_vals))
            bayes_vals = np.mean(matrix, axis=1)
            bayes_idx = int(np.argmax(bayes_vals))
            
            col_max = np.max(matrix, axis=0)
            regret_matrix = col_max - matrix
            max_regrets = np.max(regret_matrix, axis=1)
            savage_idx = int(np.argmin(max_regrets))
            
            results = {
                "wald": {"val": round(float(min_rows[wald_idx]), 2), "idx": wald_idx + 1},
                "hurwicz": {"val": round(float(hurwicz_vals[hurwicz_idx]), 2), "idx": hurwicz_idx + 1},
                "bayes": {"val": round(float(bayes_vals[bayes_idx]), 2), "idx": bayes_idx + 1},
                "savage": {"val": round(float(max_regrets[savage_idx]), 2), "idx": savage_idx + 1},
                "matrix": matrix.tolist(),
                "gamma": gamma
            }
        except Exception as e:
            results = {"error": str(e)}

    return render_template('index.html', results=results)

# Wymagane dla Vercel
app.debug = False
