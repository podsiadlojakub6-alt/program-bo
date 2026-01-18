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
            data_dict = {}
            max_row, max_col = -1, -1
            
            # Zbiera wszystkie komórki, niezależnie od tego ile ich dodasz w JS
            for key in request.form.keys():
                match = re.match(r'cell_(\d+)_(\d+)', key)
                if match:
                    r, c = map(int, match.groups())
                    val = float(request.form[key].replace(',', '.'))
                    data_dict[(r, c)] = val
                    max_row = max(max_row, r)
                    max_col = max(max_col, c)
            
            if not data_dict: return render_template('index.html', results=None)

            # Tworzenie macierzy o wymiarach jakie wysłał użytkownik
            matrix = np.zeros((max_row + 1, max_col + 1))
            for (r, c), val in data_dict.items():
                matrix[r, c] = val
            
            # Obliczenia
            minima = np.min(matrix, axis=1)
            hurwicz_vals = gamma * np.min(matrix, axis=1) + (1 - gamma) * np.max(matrix, axis=1)
            bayes_vals = np.mean(matrix, axis=1)
            regret_matrix = np.max(matrix, axis=0) - matrix
            max_regrets = np.max(regret_matrix, axis=1)
            
            results = {
                "wald": {"val": round(float(np.max(minima)), 2), "idx": int(np.argmax(minima) + 1)},
                "hurwicz": {"val": round(float(np.max(hurwicz_vals)), 2), "idx": int(np.argmax(hurwicz_vals) + 1)},
                "bayes": {"val": round(float(np.max(bayes_vals)), 2), "idx": int(np.argmax(bayes_vals) + 1)},
                "savage": {"val": round(float(np.min(max_regrets)), 2), "idx": int(np.argmin(max_regrets) + 1)}
            }
        except Exception as e:
            results = {"error": str(e)}

    return render_template('index.html', results=results)
