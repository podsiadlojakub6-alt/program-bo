from flask import Flask, render_template, request
import numpy as np
import re

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        try:
            gamma = float(request.form.get('gamma', 0.6).replace(',', '.'))
            data_dict = {}
            max_row, max_col = -1, -1
            
            # Pobieranie danych z formularza
            for key in request.form.keys():
                match = re.match(r'cell_(\d+)_(\d+)', key)
                if match:
                    r, c = map(int, match.groups())
                    val_raw = request.form[key].replace(',', '.').strip()
                    val = float(val_raw) if val_raw else 0.0
                    data_dict[(r, c)] = val
                    max_row = max(max_row, r)
                    max_col = max(max_col, c)
            
            if not data_dict:
                return render_template('index.html', results=None)

            # Tworzenie macierzy
            matrix = np.zeros((max_row + 1, max_col + 1))
            for (r, c), val in data_dict.items():
                matrix[r, c] = val
            
            # Obliczenia algorytmów
            minima = np.min(matrix, axis=1)
            maxima = np.max(matrix, axis=1)
            
            # 1. Wald
            wald_idx = np.argmax(minima)
            
            # 2. Hurwicz
            hurwicz_vals = gamma * minima + (1 - gamma) * maxima
            hurwicz_idx = np.argmax(hurwicz_vals)
            
            # 3. Bayes (Laplace)
            bayes_vals = np.mean(matrix, axis=1)
            bayes_idx = np.argmax(bayes_vals)
            
            # 4. Savage
            regret_matrix = np.max(matrix, axis=0) - matrix
            max_regrets = np.max(regret_matrix, axis=1)
            savage_idx = np.argmin(max_regrets)
            
            results = {
                "wald": {"val": round(float(minima[wald_idx]), 2), "idx": int(wald_idx + 1)},
                "hurwicz": {"val": round(float(hurwicz_vals[hurwicz_idx]), 2), "idx": int(hurwicz_idx + 1)},
                "bayes": {"val": round(float(bayes_vals[bayes_idx]), 2), "idx": int(bayes_idx + 1)},
                "savage": {"val": round(float(max_regrets[savage_idx]), 2), "idx": int(savage_idx + 1)},
                "matrix": matrix.tolist(), # Wysyłamy macierz z powrotem
                "gamma": gamma
            }
        except Exception as e:
            results = {"error": str(e)}

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
