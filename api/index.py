from flask import Flask, render_template, request
import numpy as np
import re

app = Flask(__name__, template_folder='../templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        try:
            gamma = float(request.form.get('gamma', 0.6).replace(',', '.'))
            
            # Dynamiczne wyciąganie danych z pól formularza (cell_x_y)
            data_dict = {}
            max_row = -1
            max_col = -1
            
            for key in request.form.keys():
                match = re.match(r'cell_(\d+)_(\d+)', key)
                if match:
                    r, c = map(int, match.groups())
                    data_dict[(r, c)] = float(request.form[key].replace(',', '.'))
                    max_row = max(max_row, r)
                    max_col = max(max_col, c)
            
            # Tworzenie macierzy numpy
            matrix = np.zeros((max_row + 1, max_col + 1))
            for (r, c), val in data_dict.items():
                matrix[r, c] = val
            
            # OBLICZENIA
            minima = np.min(matrix, axis=1)
            wald_idx = int(np.argmax(minima) + 1)
            
            hurwicz_vals = gamma * np.min(matrix, axis=1) + (1 - gamma) * np.max(matrix, axis=1)
            hurwicz_idx = int(np.argmax(hurwicz_vals) + 1)
            
            bayes_vals = np.mean(matrix, axis=1)
            bayes_idx = int(np.argmax(bayes_vals) + 1)
            
            max_cols = np.max(matrix, axis=0)
            regret_matrix = max_cols - matrix
            max_regrets = np.max(regret_matrix, axis=1)
            savage_idx = int(np.argmin(max_regrets) + 1)
            
            results = {
                "wald": {"val": round(float(minima[wald_idx-1]), 2), "idx": wald_idx},
                "hurwicz": {"val": round(float(hurwicz_vals[hurwicz_idx-1]), 2), "idx": hurwicz_idx},
                "bayes": {"val": round(float(bayes_vals[bayes_idx-1]), 2), "idx": bayes_idx},
                "savage": {"val": round(float(max_regrets[savage_idx-1]), 2), "idx": savage_idx}
            }
        except Exception as e:
            results = {"error": str(e)}

    return render_template('index.html', results=results)
