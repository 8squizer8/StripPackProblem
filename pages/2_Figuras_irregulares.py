# pages/2_Geométricas.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
import numpy as np
import pulp as pl
import matplotlib.pyplot as plt

# --- Configuração da Página ---
st.set_page_config(page_title="Figuras Geométricas", layout="wide")
st.title("Otimização por Retângulo Envolvente")
st.info("""
Esta página otimiza a disposição de figuras geométricas complexas (irregulares, com diagonais). 
O método consiste em calcular o "retângulo envolvente" de cada peça e otimizar a posição desses retângulos.
""")

# --- Inicialização do Session State ---
if 'geometric_pieces' not in st.session_state:
    st.session_state.geometric_pieces = []

# =============================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO E VISUALIZAÇÃO
# =============================================================================
def calculate_bounding_box(vertices):
    if not vertices:
        return {'w': 0, 'h': 0, 'min_x': 0, 'min_y': 0}
    x_coords = [v['X'] for v in vertices]; y_coords = [v['Y'] for v in vertices]
    min_x, max_x = min(x_coords), max(x_coords); min_y, max_y = min(y_coords), max(y_coords)
    w = max_x - min_x; h = max_y - min_y
    return {'w': w, 'h': h, 'min_x': min_x, 'min_y': min_y}

def plot_piece_with_bounding_box(piece):
    fig = go.Figure()
    all_x = [v['X'] for v in piece['vertices']]; all_y = [v['Y'] for v in piece['vertices']]
    if not all_x: return fig
    shape_x = all_x + [all_x[0]]; shape_y = all_y + [all_y[0]]
    fig.add_trace(go.Scatter(x=shape_x, y=shape_y, fill='toself', fillcolor='rgba(135, 206, 250, 0.7)', line=dict(color='black'), name='Peça'))
    if piece['is_confirmed'] and piece['bounding_box']:
        bb = piece['bounding_box']
        bb_x = [bb['min_x'], bb['min_x'] + bb['w'], bb['min_x'] + bb['w'], bb['min_x'], bb['min_x']]
        bb_y = [bb['min_y'], bb['min_y'], bb['min_y'] + bb['h'], bb['min_y'] + bb['h'], bb['min_y']]
        fig.add_trace(go.Scatter(x=bb_x, y=bb_y, mode='lines', line=dict(color='red', dash='dash'), name='Ret. Envolvente'))
    fig.update_layout(
        title=piece['name'],
        xaxis=dict(gridcolor='LightGray', dtick=1, showgrid=True),
        yaxis=dict(gridcolor='LightGray', dtick=1, showgrid=True, scaleanchor="x", scaleratio=1),
        showlegend=True, margin=dict(l=0, r=0, t=40, b=0), height=300
    )
    return fig

# =============================================================================
# FUNÇÃO DE OTIMIZAÇÃO (O modelo SPP da Página 1)
# =============================================================================

# --- ALTERAÇÃO AQUI: Adicionado 'time_limit' ---
def solve_rectangle_packing(rect_data, H, time_limit, W0=0.0, H0=0.0):
    rects = list(rect_data.keys())
    w_orig = {i: data['w'] for i, data in rect_data.items()}
    h_orig = {i: data['h'] for i, data in rect_data.items()}
    M = sum(w_orig.values()) + sum(h_orig.values()) + H + 10.0
    model = pl.LpProblem("Strip_Packing_Minimize_W_Geom", pl.LpMinimize)
    W = pl.LpVariable("W", lowBound=0.0)
    Xc = pl.LpVariable.dicts("Xc", rects, lowBound=0.0)
    Yc = pl.LpVariable.dicts("Yc", rects, lowBound=0.0)
    Wf = pl.LpVariable.dicts("Wf", rects, lowBound=0.0)
    Hf = pl.LpVariable.dicts("Hf", rects, lowBound=0.0)
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
                model += R[(i, j)] + R[(j, i)] + U[(i,j)] + U[(j, i)] <= 3, f"consistency_{i}_{j}"
    
    # --- ALTERAÇÃO AQUI: 'timeLimit' é usado ---
    status = model.solve(pl.PULP_CBC_CMD(msg=0, timeLimit=time_limit)) 
    return model, status

def plot_final_solution_rectangles(model, H, all_rects_data):
    fig = go.Figure()
    W_sol = model.objective.value()
    colors = plotly.colors.qualitative.Plotly
    for i, (rect_id, data) in enumerate(all_rects_data.items()):
        xc_sol = model.variablesDict()[f'Xc_{rect_id}'].varValue; yc_sol = model.variablesDict()[f'Yc_{rect_id}'].varValue
        wf_sol = model.variablesDict()[f'Wf_{rect_id}'].varValue; hf_sol = model.variablesDict()[f'Hf_{rect_id}'].varValue
        bottom_left_x = xc_sol - 0.5 * wf_sol; bottom_left_y = yc_sol - 0.5 * hf_sol
        fig.add_shape(type="rect", x0=bottom_left_x, y0=bottom_left_y, x1=bottom_left_x + wf_sol, y1=bottom_left_y + hf_sol, line=dict(color="Black", width=1.5), fillcolor=colors[i % len(colors)], opacity=0.9)
        label = f"P{rect_id}<br>({data['name']})"; fig.add_annotation(x=xc_sol, y=yc_sol, text=label, showarrow=False, font=dict(color="white", size=8))
    fig.add_shape(type="rect", x0=0, y0=0, x1=W_sol, y1=H, line=dict(color="Black", width=2), fillcolor="rgba(240, 248, 255, 0.5)", layer="below")
    fig.update_layout(
        title="Solução de Otimização (Retângulos Envolventes)",
        xaxis_title=f"Comprimento Mínimo (W) = {W_sol:.2f}", yaxis_title=f"Altura Fixa (H) = {H:.2f}",
        xaxis=dict(range=[0, W_sol], gridcolor='DarkGray', dtick=1, showgrid=True), 
        yaxis=dict(range=[0, H], gridcolor='DarkGray', dtick=1, showgrid=True, scaleanchor="x", scaleratio=1),
        margin=dict(l=40, r=40, t=60, b=40), height=600
    )
    return fig

# =============================================================================
# INTERFACE DO STREAMLIT (Adaptada para Bounding Box)
# =============================================================================
col1, col2 = st.columns([1, 2])
with col1:
    st.header("1. Adicionar Nova Peça")
    num_vertices_total = st.number_input("Número de Vértices da Peça", min_value=3, value=3, step=1, key="num_vertices_geom")
    with st.form(key="geometric_piece_form", clear_on_submit=True):
        piece_name = st.text_input("Nome da Peça", f"Peca {len(st.session_state.geometric_pieces) + 1}")
        quantity = st.number_input("Quantidade", min_value=1, value=1, step=1)
        df_rows = [{'X': 0.0, 'Y': 0.0} for _ in range(num_vertices_total)]
        df = pd.DataFrame(df_rows)
        st.write("Insira as coordenadas dos vértices da peça completa:")
        edited_df = st.data_editor(df, num_rows="fixed", column_config={"X": st.column_config.NumberColumn(format="%.1f"), "Y": st.column_config.NumberColumn(format="%.1f")})
        if st.form_submit_button("Adicionar Peça à Lista"):
            piece_data = {"name": piece_name, "quantity": quantity, "vertices": edited_df.to_dict('records'), "is_confirmed": False, "bounding_box": None}
            st.session_state.geometric_pieces.append(piece_data)
            st.success(f"Peça '{piece_name}' adicionada!")
with col2:
    st.header("2. Confirmar Peças Criadas")
    if not st.session_state.geometric_pieces:
        st.info("Nenhuma peça foi adicionada ainda.")
    else:
        for i, piece in enumerate(st.session_state.geometric_pieces):
            st.markdown("---")
            col_fig, col_tool = st.columns(2)
            with col_fig:
                st.write(f"**{piece['name']} (Qtd: {piece['quantity']})**")
                fig = plot_piece_with_bounding_box(piece)
                st.plotly_chart(fig, use_container_width=True)
            with col_tool:
                if piece['is_confirmed']:
                    st.success(f"Peça '{piece['name']}' confirmada!")
                    bb = piece['bounding_box']
                    st.write(f"Retângulo Envolvente: L={bb['w']:.2f}, A={bb['h']:.2f}")
                    if st.button("Remover Peça", key=f"remove_geom_{i}", type="secondary"):
                        st.session_state.geometric_pieces.pop(i); st.rerun()
                else:
                    with st.expander("Ferramentas de Confirmação", expanded=True):
                        st.write("A peça precisa ser confirmada para calcular o seu retângulo envolvente.")
                        if st.button("Confirmar e Calcular Ret. Envolvente", key=f"confirm_geom_{i}", type="primary"):
                            bb = calculate_bounding_box(piece['vertices']); piece['bounding_box'] = bb; piece['is_confirmed'] = True; st.rerun()

st.markdown("---")
st.header("3. Iniciar Otimização")

H_strip = st.number_input("Altura Total da Tira (H)", min_value=1.0, value=20.0, step=0.5, help="Defina a altura fixa da sua tira de material.")
# --- ALTERAÇÃO AQUI: Adicionado o input para o limite de tempo ---
time_limit = st.number_input("Limite de Tempo (segundos)", value=60, min_value=5, max_value=600, step=5, key="time_geom", help="Tempo máximo que o solver pode correr. Soluções mais rápidas podem não ser as ótimas.")

all_confirmed = all(p['is_confirmed'] for p in st.session_state.geometric_pieces) if st.session_state.geometric_pieces else False

if st.button("Otimizar Disposição das Peças", type="primary", disabled=(not all_confirmed)):
    with st.spinner("A calcular a solução..."):
        try:
            rect_data = {}
            rect_id = 1
            for piece in st.session_state.geometric_pieces:
                for _ in range(piece['quantity']):
                    rect_data[rect_id] = {'w': piece['bounding_box']['w'], 'h': piece['bounding_box']['h'], 'name': piece['name']}
                    rect_id += 1
            
            # --- ALTERAÇÃO AQUI: Passar o 'time_limit' para a função ---
            model, status = solve_rectangle_packing(rect_data, H_strip, time_limit)
            
            if status == pl.LpStatusOptimal:
                st.success("Solução ótima encontrada!")
            elif status == pl.LpStatusNotSolved:
                st.warning("O solver não iniciou (verifique os parâmetros).")
            elif status == pl.LpStatusInfeasible:
                st.error("Problema Infactível: Não é possível encaixar as peças na altura H fornecida.")
            elif status == pl.LpStatusUnbounded:
                st.error("Problema Ilimitado (Unbounded).")
            elif status == pl.LpStatusUndefined:
                st.warning("Solução não definida (pode ter atingido o limite de tempo sem encontrar uma solução).")
            
            # Mesmo que não seja "Ótima", pode ter encontrado uma "boa" solução
            if model.objective.value() is not None:
                W_final = model.objective.value()
                st.metric(label="Comprimento Mínimo da Tira (W)", value=f"{W_final:.2f}")
                solution_fig = plot_final_solution_rectangles(model, H_strip, rect_data)
                st.plotly_chart(solution_fig, use_container_width=True)
            else:
                 st.error("Nenhuma solução foi encontrada (nem mesmo uma não-ótima).")
        
        except Exception as e:
            st.error(f"Ocorreu um erro durante a otimização: {e}"); st.exception(e)