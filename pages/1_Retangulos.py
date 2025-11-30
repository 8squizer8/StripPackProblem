# app.py
import streamlit as st
import pulp as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import matplotlib

# -------------------------------------------------------------------
# FUNÇÃO 1: Resolve o problema de otimização com PuLP 
# -------------------------------------------------------------------
def solve_strip_packing(rect_data, H, W0=0.0, H0=0.0):
    """
    Esta função é uma réplica da lógica correta do seu script original 'trabalho.py'.
    Ela calcula a solução numérica ótima.
    """
    rects = list(rect_data.keys())
    w_orig = {i: data['w'] for i, data in rect_data.items()}
    h_orig = {i: data['h'] for i, data in rect_data.items()}
    M = sum(w_orig.values()) + sum(h_orig.values()) + H + 10.0

    model = pl.LpProblem("Strip_Packing_Minimize_W", pl.LpMinimize)

    W = pl.LpVariable("W", lowBound=0.0, cat="Continuous")
    Xc = pl.LpVariable.dicts("Xc", rects, lowBound=0.0, cat="Continuous")
    Yc = pl.LpVariable.dicts("Yc", rects, lowBound=0.0, cat="Continuous")
    Wf = pl.LpVariable.dicts("Wf", rects, lowBound=0.0, cat="Continuous")
    Hf = pl.LpVariable.dicts("Hf", rects, lowBound=0.0, cat="Continuous")
    V = pl.LpVariable.dicts("V", rects, cat="Binary")
    
    R, U = {}, {}
    for i in rects:
        for j in rects:
            if i != j:
                R[(i, j)] = pl.LpVariable(f"r_{i}_{j}", cat="Binary")
                U[(i, j)] = pl.LpVariable(f"u_{i}_{j}", cat="Binary")

    model += W, "Minimize_total_width"

    for i in rects:
        model += Xc[i] - 0.5 * Wf[i] >= W0, f"left_bound_{i}"
        model += Xc[i] + 0.5 * Wf[i] <= W, f"right_bound_{i}"
        model += Yc[i] - 0.5 * Hf[i] >= H0, f"bottom_bound_{i}"
        model += Yc[i] + 0.5 * Hf[i] <= H, f"top_bound_{i}"
        model += Wf[i] == (1 - V[i]) * w_orig[i] + V[i] * h_orig[i], f"Wf_def_{i}"
        model += Hf[i] == (1 - V[i]) * h_orig[i] + V[i] * w_orig[i], f"Hf_def_{i}"

    for i in rects:
        for j in rects:
            if i != j:
                model += Xc[i] + 0.5 * Wf[i] <= Xc[j] - 0.5 * Wf[j] + M * R[(i, j)], f"hor_sep_{i}_{j}"
                model += Yc[i] + 0.5 * Hf[i] <= Yc[j] - 0.5 * Hf[j] + M * U[(i, j)], f"ver_sep_{i}_{j}"

    for i in rects:
        for j in rects:
            if i < j:
                model += R[(i, j)] + R[(j, i)] + U[(i, j)] + U[(j, i)] <= 3, f"consistency_{i}_{j}"

    status = model.solve(pl.PULP_CBC_CMD(msg=0)) 
    return model, status

# -------------------------------------------------------------------
# FUNÇÃO 2: Cria a visualização gráfica da solução
# -------------------------------------------------------------------
def create_solution_plot(model, H, W0=0.0, H0=0.0):
    if model.status != pl.LpStatusOptimal:
        return None, None

    variables = model.variablesDict()
    W_sol = variables['W'].varValue
    rect_ids = sorted([int(v.name.split('_')[1]) for v in variables.values() if v.name.startswith('Xc_')])
    
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.set_title("Solução Ótima - Strip Packing")
    ax.set_xlabel(f"Comprimento Mínimo (W) = {W_sol:.2f}")
    ax.set_ylabel(f"Altura Fixa (H) = {H:.2f}")

    strip_background = patches.Rectangle((W0, H0), W_sol, H, linewidth=1.5, edgecolor='black', facecolor='azure', zorder=0)
    ax.add_patch(strip_background)
    
    color_list = ["#E63946", "#F1FAEE", "#A8DADC", "#457B9D", "#1D3557", "#E63946", "#F1FAEE", "#A8DADC", "#457B9D", "#1D3557"]
    results_data = []

    for idx, i in enumerate(rect_ids):
        xc_sol = variables[f'Xc_{i}'].varValue
        yc_sol = variables[f'Yc_{i}'].varValue
        wf_sol = variables[f'Wf_{i}'].varValue
        hf_sol = variables[f'Hf_{i}'].varValue
        v_sol = int(variables[f'V_{i}'].varValue)
        
        bottom_left_x = xc_sol - 0.5 * wf_sol
        bottom_left_y = yc_sol - 0.5 * hf_sol
        
        rect_patch = patches.Rectangle(
            (bottom_left_x, bottom_left_y), wf_sol, hf_sol,
            linewidth=1.5, edgecolor='black', facecolor=color_list[idx % len(color_list)]
        )
        ax.add_patch(rect_patch)
        
        ax.text(xc_sol, yc_sol, f"R{i}\n({wf_sol:.1f}x{hf_sol:.1f})", 
                ha='center', va='center', fontsize=9, color='black', weight='bold')
        
        results_data.append({
            "Retângulo": i,
            "Centro (Xc, Yc)": f"({xc_sol:.2f}, {yc_sol:.2f})",
            "W Final": f"{wf_sol:.2f}",
            "H Final": f"{hf_sol:.2f}",
            "Rotacionado?": "Sim" if v_sol == 1 else "Não"
        })
    
    ax.set_xlim(W0, W_sol)
    ax.set_ylim(H0, H)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return fig, pd.DataFrame(results_data)

# -------------------------------------------------------------------
# INTERFACE DO STREAMLIT 
# -------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Otimizador do Strip Packing Problem (Retângulos)")

st.sidebar.header("Parâmetros de Entrada")
H = st.sidebar.number_input("Altura da Tira (H)", value=10.0, min_value=1.0, step=0.5)
N = st.sidebar.number_input("Número de Retângulos", value=3, min_value=1, max_value=8, step=1)

# --- INÍCIO DA SECÇÃO MODIFICADA ---
rect_data = {}
st.sidebar.subheader("Dimensões dos Retângulos")
default_dims = {1: (10.0, 5.0), 2: (2.0, 3.0), 3: (3.0, 5.0)}

# Criar um 'expander' para cada retângulo para uma organização mais limpa
for i in range(1, N + 1):
    with st.sidebar.expander(f"Dados do Retângulo {i}", expanded=True):
        default_w, default_h = default_dims.get(i, (2.0, 2.0))
        
        w = st.number_input(
            label=f"Comprimento (w_{i})", 
            key=f"w{i}", 
            value=default_w, 
            min_value=0.1, 
            step=0.5, 
            format="%.1f"
        )
        h = st.number_input(
            label=f"Altura (h_{i})", 
            key=f"h{i}", 
            value=default_h, 
            min_value=0.1, 
            step=0.5, 
            format="%.1f"
        )
        
        # Validação para alertar o utilizador se um retângulo não couber
        if h > H and w > H:
             st.warning(f"Atenção: R{i} pode não caber na tira (w > H e h > H).")
        
        rect_data[i] = {'w': w, 'h': h}
# --- FIM DA SECÇÃO MODIFICADA ---

if st.sidebar.button("Otimizar Disposição", type="primary", use_container_width=True):
    with st.spinner("A calcular a solução ótima... Isto pode demorar alguns segundos..."):
        model, status = solve_strip_packing(rect_data, H)

    if status == pl.LpStatusOptimal:
        st.success("Solução ótima encontrada!")
        W_final = pl.value(model.objective)
        st.metric(label="Comprimento Mínimo da Tira (W)", value=f"{W_final:.2f}")

        solution_plot, results_df = create_solution_plot(model, H)
        
        col1, col2 = st.columns([2, 1]) 
        with col1:
            st.pyplot(solution_plot)
        with col2:
            st.subheader("Detalhes da Solução")
            st.dataframe(results_df.set_index("Retângulo"))
    else:
        st.error("Não foi possível encontrar uma solução ótima.")
        st.write("Status da Solução:", pl.LpStatus[status])
else:
    st.info("Defina os parâmetros na barra lateral e clique em 'Otimizar Disposição' para começar.")