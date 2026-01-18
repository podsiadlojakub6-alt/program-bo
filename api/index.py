from flask import Flask, render_template, request
import numpy as np
import re

app = Flask(__name__, template_folder='../templates')

def solve_games(matrix):
    """Logika dla Gier Strategicznych (Gracz A vs Gracz B)"""
    r_min = np.min(matrix, axis=1)
    c_max = np.max(matrix, axis=0)
    maximin = float(np.max(r_min))
    minimax = float(np.min(c_max))
    
    res = {
        "va": maximin, 
        "vb": minimax, 
        "equal": maximin == minimax,
        "saddle": None, 
        "mixed": None
    }
    
    # Sprawdzenie punktu siodłowego
    if maximin == minimax:
        r_idx, c_idx = int(np.argmax(r_min)), int(np.argmin(c_max))
        res["saddle"] = {"v": maximin, "r": r_idx + 1, "c": c_idx + 1}
    # Strategie mieszane (dla macierzy 2x2 bez punktu siodłowego)
    elif matrix.shape == (2, 2):
        a11, a12 = matrix[0,0], matrix[0,1]
        a21, a22 = matrix[1,0], matrix[1,1]
        den = (a11 + a22) - (a12 + a21)
        if den != 0:
            p1 = (a22 - a21) / den
            q1 = (a22 - a12) / den
            v = (a11*a22 - a12*a21) / den
            res["mixed"] = {
                "p1_pct": round(p1 * 100, 1), "p2_pct": round((1 - p1) * 100, 1),
                "q1_pct": round(q1 * 100, 1), "q2_pct": round((1 - q1) * 100, 1),
                "v": round(float(v), 2)
            }
    return res

@app.route('/', methods=['GET', 'POST'])
def index():
    results, mode = None, request.form.get('mode', 'nature')
    if request.method == 'POST':
        try:
            # Parsowanie komórek macierzy z formularza
            cells = {tuple(map(int, re.match(r'cell_(\d+)_(\d+)', k).groups())): float(v.replace(',', '.')) 
                     for k, v in request.form.items() if re.match(r'cell_(\d+)_(\d+)', k)}
            
            if not cells: 
                return render_template('index.html', results=None, mode=mode)
                
            r_m, c_m = max(k[0] for k in cells.keys()), max(k[1] for k in cells.keys())
            mat = np.zeros((r_m + 1, c_m + 1))
            for (r, c), v in cells.items(): 
                mat[r, c] = v
            
            if mode == 'nature':
                g = float(request.form.get('gamma', 0.6).replace(',', '.'))
                mi = np.min(mat, axis=1)
                # Kryteria: Wald (maximin), Hurwicz, Bayes (Laplace), Savage (minimax strat)
                results = {
                    "type": "nature", 
                    "wald": int(np.argmax(mi)+1), 
                    "hur": int(np.argmax(g*mi + (1-g)*np.max(mat, axis=1))+1),
                    "bayes": int(np.argmax(np.mean(mat, axis=1))+1), 
                    "sav": int(np.argmin(np.max(np.max(mat, axis=0)-mat, axis=1))+1)
                }
            elif mode == 'game': 
                results = {"type": "game", "data": solve_games(mat)}
                
        except Exception as e: 
            results = {"error": str(e)}
            
    return render_template('index.html', results=results, mode=mode)
