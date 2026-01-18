from flask import Flask, render_template, request
import numpy as np
import re

app = Flask(__name__, template_folder='../templates')

def oblicz_gre_macierzowa(matrix):
    # Znajdowanie punktu siodłowego
    row_mins = np.min(matrix, axis=1)
    col_maxs = np.max(matrix, axis=0)
    
    maximin = np.max(row_mins)
    minimax = np.min(col_maxs)
    
    saddle_point = None
    if maximin == minimax:
        # Znajdź współrzędne punktu siodłowego
        r_idx = int(np.argmax(row_mins))
        c_idx = int(np.argmin(col_maxs))
        saddle_point = {"val": float(maximin), "row": r_idx + 1, "col": c_idx + 1}
    
    return {
        "maximin": float(maximin),
        "minimax": float(minimax),
        "saddle_point": saddle_point,
        "is_stable": maximin == minimax
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    mode = request.form.get('mode', 'bo') # Domyślnie Badania Operacyjne (Kryteria)
    
    if request.method == 'POST':
        try:
            data_dict = {}
            max_row, max_col = -1, -1
            for key in request.form.keys():
                match = re.match(r'cell_(\d+)_(\d+)', key)
                if match:
                    r, c = map(int, match.groups())
                    data_dict[(r, c)] = float(request.form[key].replace(',', '.'))
                    max_row, max_col = max(max_row, r), max(max_col, c)
            
            if data_dict:
                matrix = np.zeros((max_row + 1, max_col + 1))
                for (r, c), val in data_dict.items(): matrix[r, c] = val

                if mode == 'bo':
                    gamma = float(request.form.get('gamma', 0.6).replace(',', '.'))
                    row_mins = np.min(matrix, axis=1)
                    hurwicz = gamma * row_mins + (1 - gamma) * np.max(matrix, axis=1)
                    bayes = np.mean(matrix, axis=1)
                    regret = np.max(matrix, axis=0) - matrix
                    max_regrets = np.max(regret, axis=1)
                    
                    results = {
                        "type": "bo",
                        "wald": {"val": round(float(np.max(row_mins)), 2), "idx": int(np.argmax(row_mins) + 1)},
                        "hurwicz": {"val": round(float(np.max(hurwicz)), 2), "idx": int(np.argmax(hurwicz) + 1)},
                        "bayes": {"val": round(float(np.max(bayes)), 2), "idx": int(np.argmax(bayes) + 1)},
                        "savage": {"val": round(float(np.min(max_regrets)), 2), "idx": int(np.argmin(max_regrets) + 1)}
                    }
                else:
                    results = {"type": "game", "data": oblicz_gre_macierzowa(matrix)}
        except Exception as e:
            results = {"error": str(e)}

    return render_template('index.html', results=results, mode=mode)
