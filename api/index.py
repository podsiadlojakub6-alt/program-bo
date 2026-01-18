from flask import Flask, render_template, request
import numpy as np
import re

app = Flask(__name__, template_folder='../templates')

def solve_simplex(c, A, b):
    # Prosta implementacja Simplex (Max)
    num_vars = len(c)
    num_constraints = len(b)
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
    tableau[:-1, :num_vars] = A
    tableau[:-1, num_vars:num_vars + num_constraints] = np.eye(num_constraints)
    tableau[:-1, -1] = b
    tableau[-1, :num_vars] = -np.array(c)
    
    for _ in range(100): # Limit iteracji
        if not np.any(tableau[-1, :-1] < 0): break
        pivot_col = np.argmin(tableau[-1, :-1])
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[tableau[:-1, pivot_col] <= 0] = np.inf
        if np.all(ratios == np.inf): break
        pivot_row = np.argmin(ratios)
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]
        for r in range(len(tableau)):
            if r != pivot_row:
                tableau[r] -= tableau[r, pivot_col] * tableau[pivot_row]
    
    x = np.zeros(num_vars)
    for i in range(num_vars):
        col = tableau[:, i]
        if np.sum(col == 1) == 1 and np.sum(col == 0) == num_constraints:
            x[i] = tableau[np.where(col == 1)[0][0], -1]
    return {"x": np.round(x, 2).tolist(), "obj": round(float(tableau[-1, -1]), 2)}

def solve_transport(costs, supply, demand):
    s, d = list(supply), list(demand)
    rows, cols = len(s), len(d)
    allocation = np.zeros((rows, cols))
    i, j = 0, 0
    while i < rows and j < cols:
        val = min(s[i], d[j])
        allocation[i, j] = val
        s[i] -= val
        d[j] -= val
        if s[i] == 0: i += 1
        else: j += 1
    return {"allocation": allocation.tolist(), "cost": float(np.sum(allocation * costs))}

def solve_games(matrix):
    row_mins = np.min(matrix, axis=1)
    col_maxs = np.max(matrix, axis=0)
    maximin, minimax = np.max(row_mins), np.min(col_maxs)
    res = {"maximin": float(maximin), "minimax": float(minimax), "saddle": None, "mixed": None}
    if maximin == minimax:
        res["saddle"] = {"v": float(maximin), "r": int(np.argmax(row_mins)+1), "c": int(np.argmin(col_maxs)+1)}
    elif matrix.shape == (2, 2):
        a, b, c, d = matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]
        den = (a + d) - (b + c)
        if den != 0:
            p1 = (d - c) / den
            q1 = (d - b) / den
            res["mixed"] = {"p1": round(p1, 3), "p2": round(1-p1, 3), "q1": round(q1, 3), "q2": round(1-q1, 3), "v": round((a*d-b*c)/den, 3)}
    return res

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    mode = request.form.get('mode', 'bo')
    if request.method == 'POST':
        try:
            cells = {tuple(map(int, re.match(r'cell_(\d+)_(\d+)', k).groups())): float(v.replace(',', '.')) 
                     for k, v in request.form.items() if re.match(r'cell_(\d+)_(\d+)', k)}
            if not cells: return render_template('index.html', results=None, mode=mode)
            r_max, c_max = max(k[0] for k in cells.keys()), max(k[1] for k in cells.keys())
            matrix = np.zeros((r_max + 1, c_max + 1))
            for (r, c), v in cells.items(): matrix[r, c] = v

            if mode == 'bo':
                g = float(request.form.get('gamma', 0.6))
                mi = np.min(matrix, axis=1)
                results = {"type": "bo", "wald": int(np.argmax(mi)+1), "hur": int(np.argmax(g*mi + (1-g)*np.max(matrix, axis=1))+1),
                           "bayes": int(np.argmax(np.mean(matrix, axis=1))+1), "sav": int(np.argmin(np.max(np.max(matrix, axis=0)-matrix, axis=1))+1)}
            elif mode == 'game': results = {"type": "game", "data": solve_games(matrix)}
            elif mode == 'transport': results = {"type": "trans", "data": solve_transport(matrix[:-1, :-1], matrix[:-1, -1], matrix[-1, :-1])}
            elif mode == 'simplex': results = {"type": "simplex", "data": solve_simplex(matrix[-1, :-1], matrix[:-1, :-1], matrix[:-1, -1])}
        except Exception as e: results = {"error": str(e)}
    return render_template('index.html', results=results, mode=mode)
