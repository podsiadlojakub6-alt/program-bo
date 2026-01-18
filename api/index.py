from flask import Flask, render_template, request
import numpy as np
import re

app = Flask(__name__, template_folder='../templates')

def solve_games(matrix):
    r_min = np.min(matrix, axis=1)
    c_max = np.max(matrix, axis=0)
    maximin, minimax = np.max(r_min), np.min(c_max)
    res = {"maximin": float(maximin), "minimax": float(minimax), "saddle": None, "mixed": None}
    
    if maximin == minimax:
        r_idx, c_idx = int(np.argmax(r_min)), int(np.argmin(c_max))
        res["saddle"] = {"v": float(maximin), "r": r_idx + 1, "c": c_idx + 1}
    elif matrix.shape == (2, 2):
        a, b, c, d = matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]
        den = (a + d) - (b + c)
        if den != 0:
            p1 = (d - c) / den
            q1 = (d - b) / den
            v = (a*d - b*c) / den
            # Tworzenie opisów "toku myślenia"
            res["mixed"] = {
                "p1_pct": round(p1 * 100, 1), "p2_pct": round((1-p1) * 100, 1),
                "q1_pct": round(q1 * 100, 1), "q2_pct": round((1-q1) * 100, 1),
                "v": round(v, 2),
                "desc_a": f"Gracz A: 'Wybieram W1 w {round(p1*100)}% i W2 w {round((1-p1)*100)}% czasu. Dzięki temu, niezależnie co zrobi przeciwnik, mój średni zysk to {round(v, 2)}.'",
                "desc_b": f"Gracz B: 'Wybieram K1 w {round(q1*100)}% i K2 w {round((1-q1)*100)}% czasu. W ten sposób ograniczam moją średnią stratę do {round(v, 2)}.'"
            }
    return res

@app.route('/', methods=['GET', 'POST'])
def index():
    results, mode = None, request.form.get('mode', 'nature')
    if request.method == 'POST':
        try:
            cells = {tuple(map(int, re.match(r'cell_(\d+)_(\d+)', k).groups())): float(v.replace(',', '.')) 
                     for k, v in request.form.items() if re.match(r'cell_(\d+)_(\d+)', k)}
            if not cells: return render_template('index.html', results=None, mode=mode)
            r_m, c_m = max(k[0] for k in cells.keys()), max(k[1] for k in cells.keys())
            mat = np.zeros((r_m + 1, c_m + 1))
            for (r, c), v in cells.items(): mat[r, c] = v
            if mode == 'nature':
                g = float(request.form.get('gamma', 0.6))
                mi = np.min(mat, axis=1)
                results = {"type": "nature", "wald": int(np.argmax(mi)+1), "hur": int(np.argmax(g*mi + (1-g)*np.max(mat, axis=1))+1),
                           "bayes": int(np.argmax(np.mean(mat, axis=1))+1), "sav": int(np.argmin(np.max(np.max(mat, axis=0)-mat, axis=1))+1)}
            elif mode == 'game': results = {"type": "game", "data": solve_games(mat)}
            # ... reszta modułów (transport/simplex) pozostaje bez zmian ...
            elif mode == 'transport':
                from api.index import solve_transport # zakładając że funkcje są w tym samym pliku
                results = {"type": "trans", "data": solve_transport(mat[:-1, :-1], mat[:-1, -1], mat[-1, :-1])}
            elif mode == 'simplex':
                from api.index import solve_simplex
                results = {"type": "simplex", "data": solve_simplex(mat[-1, :-1], mat[:-1, :-1], mat[:-1, -1])}
        except Exception as e: results = {"error": str(e)}
    return render_template('index.html', results=results, mode=mode)
