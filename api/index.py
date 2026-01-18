from flask import Flask, render_template, request
import numpy as np
import re

app = Flask(__name__, template_folder='../templates')

def solve_simplex(c, A, b):
    num_vars, num_constraints = len(c), len(b)
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
    tableau[:-1, :num_vars], tableau[:-1, num_vars:num_vars + num_constraints] = A, np.eye(num_constraints)
    tableau[:-1, -1], tableau[-1, :num_vars] = b, -np.array(c)
    for _ in range(100):
        if not np.any(tableau[-1, :-1] < 0): break
        p_col = np.argmin(tableau[-1, :-1])
        r = tableau[:-1, -1] / tableau[:-1, p_col]
        r[tableau[:-1, p_col] <= 0] = np.inf
        if np.all(r == np.inf): break
        p_row = np.argmin(r)
        tableau[p_row] /= tableau[p_row, p_col]
        for idx in range(len(tableau)):
            if idx != p_row: tableau[idx] -= tableau[idx, p_col] * tableau[p_row]
    x = np.zeros(num_vars)
    for i in range(num_vars):
        col = tableau[:, i]
        if np.sum(col == 1) == 1 and np.sum(col == 0) == num_constraints:
            x[i] = tableau[np.where(col == 1)[0][0], -1]
    return {"x": np.round(x, 2).tolist(), "obj": round(float(tableau[-1, -1]), 2)}

def solve_transport(costs, supply, demand):
    s, d = list(supply), list(demand)
    rows, cols = len(s), len(d)
    alloc = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            v = min(s[i], d[j])
            alloc[i, j] = v
            s[i], d[j] = s[i]-v, d[j]-v
    return {"alloc": alloc.tolist(), "cost": float(np.sum(alloc * costs))}

def solve_games(matrix):
    r_min = np.min(matrix, axis=1)
    c_max = np.max(matrix, axis=0)
    maximin = float(np.max(r_min))
    minimax = float(np.min(c_max))
    res = {"va": maximin, "vb": minimax, "equal": maximin == minimax, "saddle": None, "mixed": None}
    if maximin == minimax:
        r_idx, c_idx = int(np.argmax(r_min)), int(np.argmin(c_max))
        res["saddle"] = {"v": maximin, "r": r_idx + 1, "c": c_idx + 1}
    elif matrix.shape == (2, 2):
        a11, a12, a21, a22 = matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]
        den = (a11 + a22) - (a12 + a21)
        if den != 0:
            p1 = (a22 - a21) / den
            q1 = (a22 - a12) / den
            v = (a11*a22 - a12*a21) / den
            res["mixed"] = {
                "p1_pct": round(p1 * 100, 1), "p2_pct": round((1-p1) * 100, 1),
                "q1_pct": round(q1 * 100, 1), "q2_pct": round((1-q1) * 100, 1), "v": round(float(v), 2)
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
            elif mode == 'transport': results = {"type": "trans", "data": solve_transport(mat[:-1, :-1], mat[:-1, -1], mat[-1, :-1])}
            elif mode == 'simplex': results = {"type": "simplex", "data": solve_simplex(mat[-1, :-1], mat[:-1, :-1], mat[:-1, -1])}
        except Exception as e: results = {"error": str(e)}
    return render_template('index.html', results=results, mode=mode)
