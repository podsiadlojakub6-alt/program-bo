from flask import Flask, render_template, request
import numpy as np
import re

app = Flask(__name__, template_folder='../templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    active_tab = request.form.get('tab', 'natura') # Domyślnie karta "natura"
    
    if request.method == 'POST':
        try:
            # Pobieranie macierzy
            data_dict = {}
            max_row, max_col = -1, -1
            for key in request.form.keys():
                match = re.match(r'cell_(\d+)_(\d+)', key)
                if match:
                    r, c = map(int, match.groups())
                    val = float(request.form[key].replace(',', '.') or 0)
                    data_dict[(r, c)] = val
                    max_row, max_col = max(max_row, r), max(max_col, c)

            if max_row != -1:
                matrix = np.zeros((max_row + 1, max_col + 1))
                for (r, c), val in data_dict.items():
                    matrix[r, c] = val

                if active_tab == 'natura':
                    # OBLICZENIA DLA GIER Z NATURĄ
                    gamma = float(request.form.get('gamma', 0.6).replace(',', '.'))
                    min_rows = np.min(matrix, axis=1)
                    max_rows = np.max(matrix, axis=1)
                    
                    wald_idx = int(np.argmax(min_rows))
                    hurwicz_vals = gamma * min_rows + (1 - gamma) * max_rows
                    hurwicz_idx = int(np.argmax(hurwicz_vals))
                    bayes_vals = np.mean(matrix, axis=1)
                    bayes_idx = int(np.argmax(bayes_vals))
                    
                    regret = np.max(matrix, axis=0) - matrix
                    savage_max_regret = np.max(regret, axis=1)
                    savage_idx = int(np.argmin(savage_max_regret))

                    results = {
                        "type": "natura",
                        "wald": {"val": round(float(min_rows[wald_idx]), 2), "idx": wald_idx + 1},
                        "hurwicz": {"val": round(float(hurwicz_vals[hurwicz_idx]), 2), "idx": hurwicz_idx + 1},
                        "bayes": {"val": round(float(bayes_vals[bayes_idx]), 2), "idx": bayes_idx + 1},
                        "savage": {"val": round(float(np.min(savage_max_regret)), 2), "idx": savage_idx + 1},
                        "matrix": matrix.tolist()
                    }

                else:
                    # OBLICZENIA DLA GIER CZYSTYCH (Punkt Siodłowy)
                    row_mins = np.min(matrix, axis=1)
                    col_maxs = np.max(matrix, axis=0)
                    lower_val = np.max(row_mins)
                    upper_val = np.min(col_maxs)
                    
                    saddle_point = None
                    if lower_val == upper_val:
                        saddle_point = float(lower_val)

                    results = {
                        "type": "strategiczne",
                        "lower": float(lower_val),
                        "upper": float(upper_val),
                        "saddle": saddle_point,
                        "matrix": matrix.tolist()
                    }

        except Exception as e:
            results = {"error": str(e)}

    return render_template('index.html', results=results, active_tab=active_tab)
