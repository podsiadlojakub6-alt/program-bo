from flask import Flask, render_template, request
import numpy as np
import os

# Kluczowa zmiana: wskazanie folderu templates
app = Flask(__name__, template_folder='../templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        try:
            raw_data = request.form.get('matrix')
            gamma = float(request.form.get('gamma', 0.6).replace(',', '.'))
            
            rows = raw_data.strip().split('\n')
            matrix = [list(map(lambda x: float(x.replace(',', '.')), r.split())) for r in rows]
            data = np.array(matrix)
            
            minima = np.min(data, axis=1)
            wald_idx = int(np.argmax(minima) + 1)
            
            hurwicz_vals = gamma * np.min(data, axis=1) + (1 - gamma) * np.max(data, axis=1)
            hurwicz_idx = int(np.argmax(hurwicz_vals) + 1)
            
            bayes_vals = np.mean(data, axis=1)
            bayes_idx = int(np.argmax(bayes_vals) + 1)
            
            max_cols = np.max(data, axis=0)
            regret_matrix = max_cols - data
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

# To jest ważne dla Vercel
app.debug = False
