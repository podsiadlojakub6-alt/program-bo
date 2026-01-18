from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        try:
            # Pobieranie danych z formularza
            raw_data = request.form.get('matrix')
            gamma = float(request.form.get('gamma', 0.6).replace(',', '.'))
            
            # Przetwarzanie tekstu na macierz numpy
            rows = raw_data.strip().split('\n')
            matrix = [list(map(lambda x: float(x.replace(',', '.')), r.split())) for r in rows]
            data = np.array(matrix)
            
            # OBLICZENIA
            # 1. Wald
            minima = np.min(data, axis=1)
            wald_idx = np.argmax(minima) + 1
            
            # 2. Hurwicz
            hurwicz_vals = gamma * np.min(data, axis=1) + (1 - gamma) * np.max(data, axis=1)
            hurwicz_idx = np.argmax(hurwicz_vals) + 1
            
            # 3. Bayes (A - równe szanse)
            bayes_vals = np.mean(data, axis=1)
            bayes_idx = np.argmax(bayes_vals) + 1
            
            # 4. Savage
            max_cols = np.max(data, axis=0)
            regret_matrix = max_cols - data
            max_regrets = np.max(regret_matrix, axis=1)
            savage_idx = np.argmin(max_regrets) + 1
            
            results = {
                "wald": {"val": round(minima[wald_idx-1], 2), "idx": wald_idx},
                "hurwicz": {"val": round(hurwicz_vals[hurwicz_idx-1], 2), "idx": hurwicz_idx},
                "bayes": {"val": round(bayes_vals[bayes_idx-1], 2), "idx": bayes_idx},
                "savage": {"val": round(max_regrets[savage_idx-1], 2), "idx": savage_idx}
            }
        except Exception as e:
            results = {"error": str(e)}

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)