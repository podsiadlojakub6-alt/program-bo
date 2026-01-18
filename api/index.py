from flask import Flask, render_template, request
import numpy as np
import re

app = Flask(__name__) # Uproszczone, jeśli templates są obok pliku .py

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        try:
            # Pobieranie współczynnika gamma (jako współczynnik pesymizmu)
            gamma = float(request.form.get('gamma', 0.6).replace(',', '.'))
            
            data_dict = {}
            max_row, max_col = -1, -1
            
            # Parsowanie danych wejściowych
            for key in request.form.keys():
                match = re.match(r'cell_(\d+)_(\d+)', key)
                if match:
                    r, c = map(int, match.groups())
                    raw_val = request.form[key].strip().replace(',', '.')
                    if raw_val == "": continue # Ignoruj puste komórki
                    
                    val = float(raw_val)
                    data_dict[(r, c)] = val
                    max_row = max(max_row, r)
                    max_col = max(max_col, c)
            
            if not data_dict: 
                return render_template('index.html', results=None)

            # Budowa macierzy (Wiersze: Decyzje, Kolumny: Stany Natury)
            matrix = np.zeros((max_row + 1, max_col + 1))
            for (r, c), val in data_dict.items():
                matrix[r, c] = val
            
            # 1. Wald (Kryterium pesymistyczne - max z minimów)
            minima = np.min(matrix, axis=1)
            wald_idx = np.argmax(minima)
            
            # 2. Hurwicz (Wartość pośrednia)
            # Przyjmując gamma jako wagę dla najgorszego wyniku:
            hurwicz_vals = gamma * np.min(matrix, axis=1) + (1 - gamma) * np.max(matrix, axis=1)
            hurwicz_idx = np.argmax(hurwicz_vals)
            
            # 3. Bayes (Kryterium równego prawdopodobieństwa / Laplace'a)
            # Zakładamy, że prawdopodobieństwa stanów natury są równe
            bayes_vals = np.mean(matrix, axis=1)
            bayes_idx = np.argmax(bayes_vals)
            
            # 4. Savage (Kryterium minimalizacji żalu)
            # Macierz żalu: najlepszy wynik w kolumnie minus wynik rzeczywisty
            regret_matrix = np.max(matrix, axis=0) - matrix
            max_regrets = np.max(regret_matrix, axis=1)
            savage_idx = np.argmin(max_regrets)
            
            results = {
                "wald": {"val": round(float(minima[wald_idx]), 2), "idx": int(wald_idx + 1)},
                "hurwicz": {"val": round(float(hurwicz_vals[hurwicz_idx]), 2), "idx": int(hurwicz_idx + 1)},
                "bayes": {"val": round(float(bayes_vals[bayes_idx]), 2), "idx": int(bayes_idx + 1)},
                "savage": {"val": round(float(max_regrets[savage_idx]), 2), "idx": int(savage_idx + 1)}
            }
            
        except Exception as e:
            results = {"error": f"Błąd obliczeń: {str(e)}"}

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
