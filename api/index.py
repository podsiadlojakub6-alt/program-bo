from flask import Flask, render_template, request
import numpy as np
import re
import traceback

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        try:
            # 1. Pobranie Gammy
            gamma_raw = request.form.get('gamma', '0.6').replace(',', '.')
            gamma = float(gamma_raw) if gamma_raw else 0.6
            
            data_dict = {}
            max_row, max_col = -1, -1
            
            # 2. Bezpieczne zbieranie danych z pól cell_x_y
            for key in request.form.keys():
                match = re.match(r'cell_(\d+)_(\d+)', key)
                if match:
                    r, c = map(int, match.groups())
                    val_raw = request.form[key].replace(',', '.').strip()
                    # Jeśli pole jest puste, przyjmij 0, aby uniknąć błędu float()
                    val = float(val_raw) if val_raw else 0.0
                    
                    data_dict[(r, c)] = val
                    max_row = max(max_row, r)
                    max_col = max(max_col, c)
            
            # Jeśli macierz jest pusta, nie rób nic
            if max_row == -1 or max_col == -1:
                return render_template('index.html', results={"error": "Macierz jest pusta!"})

            # 3. Inicjalizacja macierzy o poprawnym rozmiarze
            matrix = np.zeros((max_row + 1, max_col + 1))
            for (r, c), val in data_dict.items():
                matrix[r, c] = val
            
            # 4. Obliczenia (axis=1 to wiersze - decyzje A)
            minima = np.min(matrix, axis=1)
            maxima = np.max(matrix, axis=1)
            
            # Wald
            wald_idx = int(np.argmax(minima))
            
            # Hurwicz
            hurwicz_vals = gamma * minima + (1 - gamma) * maxima
            hurwicz_idx = int(np.argmax(hurwicz_vals))
            
            # Bayes (Laplace)
            bayes_vals = np.mean(matrix, axis=1)
            bayes_idx = int(np.argmax(bayes_vals))
            
            # Savage
            col_maxima = np.max(matrix, axis=0)
            regret_matrix = col_maxima - matrix
            max_regrets = np.max(regret_matrix, axis=1)
            savage_idx = int(np.argmin(max_regrets))
            
            results = {
                "wald": {"val": round(float(minima[wald_idx]), 2), "idx": wald_idx + 1},
                "hurwicz": {"val": round(float(hurwicz_vals[hurwicz_idx]), 2), "idx": hurwicz_idx + 1},
                "bayes": {"val": round(float(bayes_vals[bayes_idx]), 2), "idx": bayes_idx + 1},
                "savage": {"val": round(float(max_regrets[savage_idx]), 2), "idx": savage_idx + 1},
                "matrix": matrix.tolist(),
                "gamma": gamma
            }

        except Exception as e:
            # Wyświetl pełny błąd w konsoli, żeby wiedzieć co się stało
            print(traceback.format_exc())
            results = {"error": f"Błąd danych: Upewnij się, że wpisano tylko liczby. ({str(e)})"}

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
