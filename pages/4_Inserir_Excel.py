# pages/4_Inserir_Excel.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
import numpy as np
import pulp as pl
import re
from io import BytesIO
import gurobipy as gp # <-- Agora vai funcionar!

# --- Configuração da Página ---
st.set_page_config(page_title="Importar Excel", layout="wide")
st.title("Importar Pedido via Excel (com Decomposição Automática)")
st.info("""
Carregue um ficheiro Excel (.xlsx) com peças ortogonais.
A aplicação irá decompor automaticamente cada peça em "Mãe" e "Filha(s)" e, de seguida, otimizar a sua disposição.
""")

# --- Inicialização do Session State ---
if 'excel_pieces_auto' not in st.session_state:
    st.session_state.excel_pieces_auto = {
        "pieces": [],
        "confirmed": False
    }

# =============================================================================
# ALGORITMO DE DECOMPOSIÇÃO AUTOMÁTICA
# (Não alterado)
# =============================================================================

def auto_decompose_polygon(vertices):
    rects = []
    y_levels = sorted(list(set(v['Y'] for v in vertices)))
    for i in range(len(y_levels) - 1):
        y_min, y_max = y_levels[i], y_levels[i+1]
        vertical_edges = []
        for j in range(len(vertices)):
            v1 = vertices[j]; v2 = vertices[(j + 1) % len(vertices)]
            if v1['X'] == v2['X'] and min(v1['Y'], v2['Y']) <= y_min and max(v1['Y'], v2['Y']) >= y_max:
                vertical_edges.append(v1['X'])
        vertical_edges = sorted(list(set(vertical_edges)))
        for j in range(len(vertical_edges) - 1):
            x_min, x_max = vertical_edges[j], vertical_edges[j+1]
            rects.append({'x': x_min, 'y': y_min, 'w': x_max - x_min, 'h': y_max - y_min})
    return rects

def assign_mae_filha(rects):
    if not rects: return {'mae': [], 'filhas': []}
    rect_props = []
    for r in rects:
        area = r['w'] * r['h']; center_x = r['x'] + r['w'] / 2; center_y = r['y'] + r['h'] / 2
        rect_props.append({'w': r['w'], 'h': r['h'], 'center_x': center_x, 'center_y': center_y, 'area': area, 'orig_rect': r})
    rect_props.sort(key=lambda r: r['area'], reverse=True)
    mae_props = rect_props[0]
    mae_data = {'w_orig': mae_props['w'], 'h_orig': mae_props['h'], 'rect_verts': calculate_rect_properties_from_dims(mae_props['w'], mae_props['h'], 0, 0)}
    filhas_data = []
    for f_props in rect_props[1:]:
        D_x = f_props['center_x'] - mae_props['center_x']; D_y = f_props['center_y'] - mae_props['center_y']
        filhas_data.append({'w_orig': f_props['w'], 'h_orig': f_props['h'], 'Dx': D_x, 'Dy': D_y, 'rect_verts': calculate_rect_properties_from_dims(f_props['w'], f_props['h'], D_x, D_y)})
    return {'mae': mae_data, 'filhas': filhas_data}

def parse_excel_sheet(df):
    try:
        data_list = [str(row[0]) for row in df.values.tolist() if pd.notna(row[0])]
        parsed_pieces = []
        i = 0
        while i < len(data_list):
            line = data_list[i].strip()
            if line.startswith("PIECE"):
                piece_name = line; quantity = 0; num_vertices = 0; vertices = []
                i += 1
                if i < len(data_list) and data_list[i].strip() == "QUANTITY": i += 1; quantity = int(data_list[i].strip())
                i += 1
                if i < len(data_list) and data_list[i].strip() == "NUMBER OF VERTICES": i += 1; num_vertices = int(data_list[i].strip())
                i += 1
                if i < len(data_list) and "VERTICES" in data_list[i]:
                    i += 1
                    for _ in range(num_vertices):
                        if i >= len(data_list): break
                        coords = re.findall(r"[-+]?\d*\.\d+|\d+", data_list[i])
                        if len(coords) == 2: vertices.append({'X': float(coords[0]), 'Y': float(coords[1])})
                        i += 1
                if piece_name and quantity > 0 and len(vertices) == num_vertices:
                    decomposed_rects = auto_decompose_polygon(vertices)
                    decomposition_data = assign_mae_filha(decomposed_rects)
                    parsed_pieces.append({"name": piece_name, "quantity": quantity, "vertices_orig": vertices, "is_confirmed": False, "decomposition": decomposition_data})
                else:
                    while i < len(data_list) and not data_list[i].startswith("PIECE"): i += 1
                    continue 
            else: i += 1 
        return parsed_pieces
    except Exception as e:
        st.error(f"Erro ao processar o Excel. Verifique o formato. Detalhe: {e}"); st.exception(e); return []

# =============================================================================
# FUNÇÕES DE OTIMIZAÇÃO (Solver e Pré-processador)
# =============================================================================

def preprocess_data_for_solver(st_pieces):
    pieces_data = []
    piece_counter = 1 
    for st_piece in st_pieces:
        for _ in range(st_piece['quantity']):
            mae_data = st_piece['decomposition']['mae']; filhas_data = st_piece['decomposition']['filhas']
            pieces_data.append({'id': piece_counter, 'name': st_piece['name'], 'mae': {'w_orig': mae_data['w_orig'], 'h_orig': mae_data['h_orig']}, 'filhas': filhas_data})
            piece_counter += 1
    return pieces_data

def solve_orthogonal_packing(pieces_data, H, time_limit, W0=0.0, H0=0.0):
    
    # --- ALTERAÇÃO AQUI (Baseado no Gurobi AI, Step 3 & 4) ---
    # 1. Credenciais WLS (do seu ficheiro gurobi.lic)
    WLS_ACCESS_ID = "b04e9716-989f-4118-b69c-32974938514e"
    WLS_SECRET = "c369d181-29b4-4b26-b3e1-9c2f480fdb32"
    LICENSE_ID = 2737454 # <-- A chave que faltava
    
    try:
        # 2. Criar o ambiente Gurobi licenciado (moderno)
        st.info(f"A ligar ao Gurobi WLS (LicenseID: {LICENSE_ID})...")
        wls_options = {
            "WLSACCESSID": WLS_ACCESS_ID,
            "WLSSECRET": WLS_SECRET,
            "LICENSEID": LICENSE_ID, 
        }
        # Isto vai funcionar agora que o gurobipy está atualizado
        env = gp.Env(params=wls_options)
        
    except gp.GurobiError as e:
        st.error(f"Erro ao carregar a licença Gurobi WLS: {e}")
        return None, "License Error", None
    except Exception as e:
        st.error(f"Erro inesperado ao criar o ambiente Gurobi: {e}")
        return None, "License Error", None
    # --- Fim da Alteração ---

    model = pl.LpProblem("Orthogonal_Item_Packing", pl.LpMinimize)
    piece_ids = [p['id'] for p in pieces_data]; all_rects = {} 
    rect_counter = 1
    for p in pieces_data:
        p['mae']['rect_id'] = f"r{rect_counter}"; p['mae']['piece_id'] = p['id']; all_rects[f"r{rect_counter}"] = p['mae']; rect_counter += 1
        for f in p['filhas']:
            f['rect_id'] = f"r{rect_counter}"; f['piece_id'] = p['id']; all_rects[f"r{rect_counter}"] = f; rect_counter += 1
    M = sum(r['w_orig'] for r in all_rects.values()) + sum(r['h_orig'] for r in all_rects.values()) + H + 1000
    W = pl.LpVariable("W", lowBound=0.0)
    Xc_mae = pl.LpVariable.dicts("Xc_mae", piece_ids, lowBound=0.0)
    Yc_mae = pl.LpVariable.dicts("Yc_mae", piece_ids, lowBound=0.0)
    z = pl.LpVariable.dicts("z", [(k, m) for k in piece_ids for m in range(1, 5)], cat="Binary")
    Wf = pl.LpVariable.dicts("Wf", all_rects.keys(), lowBound=0.0); Hf = pl.LpVariable.dicts("Hf", all_rects.keys(), lowBound=0.0)
    Xc = pl.LpVariable.dicts("Xc", all_rects.keys(), lowBound=0.0); Yc = pl.LpVariable.dicts("Yc", all_rects.keys(), lowBound=0.0)
    rect_pairs = [(id1, id2) for id1 in all_rects for id2 in all_rects if id1 < id2 and all_rects[id1]['piece_id'] != all_rects[id2]['piece_id']]
    R_vars = [(id1, id2) for id1, id2 in rect_pairs] + [(id2, id1) for id1, id2 in rect_pairs]
    U_vars = [(id1, id2) for id1, id2 in rect_pairs] + [(id2, id1) for id1, id2 in rect_pairs]
    R = pl.LpVariable.dicts("R", R_vars, cat="Binary"); U = pl.LpVariable.dicts("U", U_vars, cat="Binary")
    model += W, "Minimize_Total_Width"
    for p in pieces_data:
        k = p['id']; mae = p['mae']; mae_id = mae['rect_id']
        model += pl.lpSum(z[k, m] for m in range(1, 5)) == 1, f"unique_rot_{k}"
        model += Wf[mae_id] == (z[k,1] + z[k,3]) * mae['w_orig'] + (z[k,2] + z[k,4]) * mae['h_orig'], f"Wf_mae_{k}"
        model += Hf[mae_id] == (z[k,1] + z[k,3]) * mae['h_orig'] + (z[k,2] + z[k,4]) * mae['w_orig'], f"Hf_mae_{k}"
        model += Xc[mae_id] == Xc_mae[k]; model += Yc[mae_id] == Yc_mae[k]
        for f in p['filhas']:
            f_id = f['rect_id']
            model += Wf[f_id] == (z[k,1] + z[k,3]) * f['w_orig'] + (z[k,2] + z[k,4]) * f['h_orig'], f"Wf_filha_{f_id}"
            model += Hf[f_id] == (z[k,1] + z[k,3]) * f['h_orig'] + (z[k,2] + z[k,4]) * f['w_orig'], f"Hf_filha_{f_id}"
            delta_x = z[k,1]*f['Dx'] + z[k,2]*f['Dy'] - z[k,3]*f['Dx'] - z[k,4]*f['Dy']
            delta_y = z[k,1]*f['Dy'] - z[k,2]*f['Dx'] - z[k,3]*f['Dy'] + z[k,4]*f['Dx']
            model += Xc[f_id] == Xc[mae_id] + delta_x, f"Xc_filha_{f_id}"
            model += Yc[f_id] == Yc[mae_id] + delta_y, f"Yc_filha_{f_id}"
    for rect_id in all_rects.keys():
        model += Xc[rect_id] - 0.5 * Wf[rect_id] >= W0, f"left_bound_{rect_id}"
        model += Xc[rect_id] + 0.5 * Wf[rect_id] <= W,  f"right_bound_{rect_id}"
        model += Yc[rect_id] - 0.5 * Hf[rect_id] >= H0, f"bottom_bound_{rect_id}"
        model += Yc[rect_id] + 0.5 * Hf[rect_id] <= H,  f"top_bound_{rect_id}"
    for id1, id2 in rect_pairs:
        model += Xc[id1] + 0.5 * Wf[id1] <= Xc[id2] - 0.5 * Wf[id2] + M * R[(id1, id2)]
        model += Xc[id2] + 0.5 * Wf[id2] <= Xc[id1] - 0.5 * Wf[id1] + M * R[(id2, id1)]
        model += Yc[id1] + 0.5 * Hf[id1] <= Yc[id2] - 0.5 * Hf[id2] + M * U[(id1, id2)]
        model += Yc[id2] + 0.5 * Hf[id2] <= Yc[id1] - 0.5 * Hf[id1] + M * U[(id2, id1)]
        model += R[(id1, id2)] + R[(id2, id1)] + U[(id1, id2)] + U[(id2, id1)] <= 3
    
    try:
        # --- ALTERAÇÃO AQUI: Passar o 'env' licenciado para o PuLP ---
        status = model.solve(pl.GUROBI(
            env=env, # Passa o ambiente que criámos
            msg=0, 
            timeLimit=time_limit
        ))
    except (pl.PulpSolverError, gp.GurobiError) as e:
        # Fallback para o CBC se o Gurobi WLS falhar
        st.warning(f"Falha ao usar o Gurobi WLS ({e}).")
        st.warning("A reverter para o solver CBC (mais lento). Isto pode demorar...")
        status = model.solve(pl.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
        
    return model, status, all_rects

# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# (Não alterado)
# =============================================================================
def calculate_rect_properties_from_dims(w, h, dx, dy):
    half_w, half_h = w / 2, h / 2; center_x, center_y = dx, dy
    verts = [{'X': center_x - half_w, 'Y': center_y - half_h}, {'X': center_x + half_w, 'Y': center_y - half_h}, {'X': center_x + half_w, 'Y': center_y + half_h}, {'X': center_x - half_w, 'Y': center_y + half_h}]
    return verts

def plot_piece_from_decomp(piece):
    fig = go.Figure()
    all_x = [v['X'] for v in piece['vertices_orig']]; all_y = [v['Y'] for v in piece['vertices_orig']]
    outline_x = all_x + [all_x[0]]; outline_y = all_y + [all_x[0]]
    fig.add_trace(go.Scatter(x=outline_x, y=outline_y, mode='lines', line=dict(color='gray', dash='dash'), name='Contorno'))
    mae = piece['decomposition']['mae']; mae_verts = mae['rect_verts'] + [mae['rect_verts'][0]]
    fig.add_trace(go.Scatter(x=[v['X'] for v in mae_verts], y=[v['Y'] for v in mae_verts], fill='toself', fillcolor='rgba(65, 105, 225, 0.8)', line=dict(color='black'), name='Mãe'))
    for i, filha in enumerate(piece['decomposition']['filhas']):
        f_verts = filha['rect_verts'] + [filha['rect_verts'][0]]
        fig.add_trace(go.Scatter(x=[v['X'] for v in f_verts], y=[v['Y'] for v in f_verts], fill='toself', fillcolor='rgba(144, 238, 144, 0.8)', line=dict(color='black'), name=f'Filha {i+1}'))
    fig.update_layout(title=f"{piece['name']} (Qtd: {piece['quantity']})", xaxis=dict(gridcolor='LightGray', dtick=1, showgrid=True), yaxis=dict(gridcolor='LightGray', dtick=1, showgrid=True, scaleanchor="x", scaleratio=1), showlegend=True, margin=dict(l=0, r=0, t=40, b=0), height=300)
    return fig

def plot_final_solution(model, H, all_rects_data):
    fig = go.Figure()
    W_sol = model.objective.value()
    piece_ids = sorted(list(set(r['piece_id'] for r in all_rects_data.values())))
    colors = plotly.colors.qualitative.Plotly
    color_map = {pid: colors[i % len(colors)] for i, pid in enumerate(piece_ids)}
    for rect_id, data in all_rects_data.items():
        xc_sol = model.variablesDict()[f'Xc_{rect_id}'].varValue; yc_sol = model.variablesDict()[f'Yc_{rect_id}'].varValue
        wf_sol = model.variablesDict()[f'Wf_{rect_id}'].varValue; hf_sol = model.variablesDict()[f'Hf_{rect_id}'].varValue
        bottom_left_x = xc_sol - 0.5 * wf_sol; bottom_left_y = yc_sol - 0.5 * hf_sol
        fig.add_shape(type="rect", x0=bottom_left_x, y0=bottom_left_y, x1=bottom_left_x + wf_sol, y1=bottom_left_y + hf_sol, line=dict(color="Black", width=1.5), fillcolor=color_map[data['piece_id']], opacity=0.9)
        label = f"P{data['piece_id']}<br>({rect_id})"; fig.add_annotation(x=xc_sol, y=yc_sol, text=label, showarrow=False, font=dict(color="white", size=8))
    fig.add_shape(type="rect", x0=0, y0=0, x1=W_sol, y1=H, line=dict(color="Black", width=2), fillcolor="rgba(240, 248, 255, 0.5)", layer="below")
    fig.update_layout(
        title="Solução de Otimização Final", xaxis_title=f"Comprimento Mínimo (W) = {W_sol:.2f}", yaxis_title=f"Altura Fixa (H) = {H:.2f}",
        xaxis=dict(range=[0, W_sol], gridcolor='DarkGray', dtick=1, showgrid=True), yaxis=dict(range=[0, H], gridcolor='DarkGray', dtick=1, showgrid=True, scaleanchor="x", scaleratio=1),
        margin=dict(l=40, r=40, t=60, b=40), height=600
    )
    return fig

# =============================================================================
# INTERFACE DO STREAMLIT
# (Não alterado)
# =============================================================================

st.header("1. Carregar Ficheiro Excel")
uploaded_file = st.file_uploader("Carregue o seu ficheiro Excel de peças", type=["xlsx"])

if uploaded_file:
    excel_obj = pd.ExcelFile(uploaded_file)
    sheet_options = excel_obj.sheet_names
    selected_sheet = st.selectbox("Selecione a folha (pedido) a processar:", sheet_options)
    
    if st.button("Carregar e Processar Peças", type="primary"):
        with st.spinner(f"A processar a folha '{selected_sheet}' e a decompor peças..."):
            df = pd.read_excel(excel_obj, sheet_name=selected_sheet, header=None)
            parsed_pieces = parse_excel_sheet(df)
            st.session_state.excel_pieces_auto["pieces"] = parsed_pieces
            st.session_state.excel_pieces_auto["confirmed"] = False
            if parsed_pieces: st.success(f"Foram carregadas e decompostas {len(parsed_pieces)} peças!")
            else: st.error("Nenhuma peça foi encontrada ou o formato está incorreto.")

if st.session_state.excel_pieces_auto["pieces"]:
    st.header("2. Rever Peças Processadas")
    pieces = st.session_state.excel_pieces_auto["pieces"]
    num_pieces = len(pieces)
    display_cols = st.columns(min(num_pieces, 4))
    
    indices_to_remove = []
    for i, piece in enumerate(pieces):
        with display_cols[i % 4]:
            fig = plot_piece_from_decomp(piece)
            st.plotly_chart(fig, use_container_width=True)
            if st.button(f"Remover {piece['name']}", key=f"remove_auto_{i}", type="secondary"):
                indices_to_remove.append(i) 
                st.session_state.excel_pieces_auto["confirmed"] = False
    
    if indices_to_remove:
        st.session_state.excel_pieces_auto["pieces"] = [p for i, p in enumerate(pieces) if i not in indices_to_remove]
        st.rerun()
            
    if not st.session_state.excel_pieces_auto["confirmed"]:
        if st.button("Confirmar Lote de Peças", key="confirm_batch"):
            st.session_state.excel_pieces_auto["confirmed"] = True
            st.rerun()
    else:
        st.success("Lote de peças confirmado!")

st.markdown("---")
st.header("3. Iniciar Otimização")

H_strip = st.number_input("Altura Total da Tira (H)", min_value=1.0, value=20.0, step=0.5, help="Defina a altura fixa da sua tira de material.")
time_limit = st.number_input("Limite de Tempo (segundos)", value=60, min_value=5, max_value=600, step=5, key="time_excel", help="Tempo máximo que o solver pode correr. Soluções mais rápidas podem não ser as ótimas.")

all_confirmed = st.session_state.excel_pieces_auto["confirmed"]

if st.button("Otimizar Disposição das Peças", type="primary", disabled=(not all_confirmed)):
    with st.spinner(f"A calcular a solução... (Solver: GUROBI WLS | Limite: {time_limit}s)"):
        try:
            processed_data = preprocess_data_for_solver(st.session_state.excel_pieces_auto["pieces"])
            model, status, all_rects_data = solve_orthogonal_packing(processed_data, H_strip, time_limit)
            
            # --- LÓGICA DE STATUS ROBUSTA (CORRIGIDA) ---
            if status == "License Error":
                pass
            else:
                status_str = pl.LpStatus[status]
                
                if status == pl.LpStatusOptimal:
                    st.success("Solução ótima encontrada!")
                elif status == pl.LpStatusInfeasible:
                    st.error("Erro: Problema Infactível. Não é possível encaixar as peças na altura H fornecida. Tente um H maior.")
                
                if model.objective is not None and model.objective.value() is not None:
                    if status != pl.LpStatusOptimal:
                        st.warning(f"Solução PARCIAL encontrada (pode não ser a ótima). O solver parou com status: {status_str}")
                    
                    W_final = model.objective.value()
                    st.metric(label="Melhor Comprimento Encontrado (W)", value=f"{W_final:.2f}")
                    solution_fig = plot_final_solution(model, H_strip, all_rects_data)
                    st.plotly_chart(solution_fig, use_container_width=True)
                
                elif status == pl.LpStatusUndefined:
                    st.error(f"Nenhuma solução foi encontrada. O solver parou com status: {status_str}. Tente aumentar o limite de tempo.")
                
                elif status not in (pl.LpStatusOptimal, pl.LpStatusInfeasible):
                     st.error(f"O solver falhou em encontrar uma solução. Status: {status_str}")

        except Exception as e:
            st.error(f"Ocorreu um erro durante a otimização: {e}"); st.exception(e)