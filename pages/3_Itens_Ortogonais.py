# pages/3_Itens_Ortogonais.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
import numpy as np
import pulp as pl
import re
from io import BytesIO

# --- Configuração da Página ---
st.set_page_config(page_title="Itens Ortogonais", layout="wide")
st.title("Definição e Otimização de Itens Ortogonais")

# --- Inicialização do Session State ---
if 'orthogonal_pieces' not in st.session_state:
    st.session_state.orthogonal_pieces = []

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO E OTIMIZAÇÃO
# =============================================================================
def calculate_rect_properties(vertices):
    if not vertices: return {'w': 0, 'h': 0, 'center_x': 0, 'center_y': 0}
    x_coords = [v['X'] for v in vertices]; y_coords = [v['Y'] for v in vertices]
    min_x, max_x = min(x_coords), max(x_coords); min_y, max_y = min(y_coords), max(y_coords)
    w = max_x - min_x; h = max_y - min_y
    center_x = min_x + w / 2; center_y = min_y + h / 2
    return {'w': w, 'h': h, 'center_x': center_x, 'center_y': center_y}

def preprocess_data_for_solver(st_pieces):
    pieces_data = []
    piece_counter = 1 
    for st_piece in st_pieces:
        for _ in range(st_piece['quantity']):
            mae_props = calculate_rect_properties(st_piece['decomposition']['mae'])
            filhas_data = []
            for filha_verts in st_piece['decomposition']['filhas']:
                filha_props = calculate_rect_properties(filha_verts)
                D_x = filha_props['center_x'] - mae_props['center_x']
                D_y = filha_props['center_y'] - mae_props['center_y']
                filhas_data.append({'w_orig': filha_props['w'], 'h_orig': filha_props['h'], 'Dx': D_x, 'Dy': D_y})
            pieces_data.append({'id': piece_counter, 'name': st_piece['name'], 'mae': {'w_orig': mae_props['w'], 'h_orig': mae_props['h']}, 'filhas': filhas_data})
            piece_counter += 1
    return pieces_data

# --- FUNÇÃO DE OTIMIZAÇÃO (CBC) ---
def solve_orthogonal_packing(pieces_data, H, time_limit, W0=0.0, H0=0.0):
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
    
    status = model.solve(pl.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    return model, status, all_rects

# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================
def plot_piece(piece):
    fig = go.Figure()
    
    # Desenhar contorno se existir (peça original)
    if piece.get('vertices'):
        all_x = [v['X'] for v in piece['vertices']]; all_y = [v['Y'] for v in piece['vertices']]
        if all_x: # Verificar se não está vazio
            outline_x_coords = all_x + [all_x[0]]; outline_y_coords = all_y + [all_y[0]]
            fig.add_trace(go.Scatter(x=outline_x_coords, y=outline_y_coords, mode='lines', line=dict(color='gray', dash='dash'), name='Contorno'))
    
    decomp = piece['decomposition']
    
    # Desenhar Mãe
    if decomp and decomp.get('mae'):
        mae_verts_list = decomp['mae'] + [decomp['mae'][0]]
        mae_x = [v['X'] for v in mae_verts_list]; mae_y = [v['Y'] for v in mae_verts_list]
        fig.add_trace(go.Scatter(x=mae_x, y=mae_y, fill='toself', fillcolor='rgba(65, 105, 225, 0.8)', line=dict(color='black'), name='Mãe'))
    
    # Desenhar Filhas
    if decomp and decomp.get('filhas'):
        for i, filha_verts in enumerate(decomp['filhas']):
            if filha_verts:
                filha_verts_list = filha_verts + [filha_verts[0]]
                f_x = [v['X'] for v in filha_verts_list]; f_y = [v['Y'] for v in filha_verts_list]
                fig.add_trace(go.Scatter(x=f_x, y=f_y, fill='toself', fillcolor='rgba(144, 238, 144, 0.8)', line=dict(color='black'), name=f'Filha {i+1}'))
    
    fig.update_layout(title=piece['name'], xaxis=dict(gridcolor='LightGray', dtick=1, showgrid=True), yaxis=dict(gridcolor='LightGray', dtick=1, showgrid=True, scaleanchor="x", scaleratio=1), margin=dict(l=0, r=0, t=40, b=0), height=300)
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
    fig.update_layout(title="Solução de Otimização Final", xaxis_title=f"Comprimento Mínimo (W) = {W_sol:.2f}", yaxis_title=f"Altura Fixa (H) = {H:.2f}", xaxis=dict(range=[0, W_sol], gridcolor='DarkGray', dtick=1, showgrid=True), yaxis=dict(range=[0, H], gridcolor='DarkGray', dtick=1, showgrid=True, scaleanchor="x", scaleratio=1), margin=dict(l=40, r=40, t=60, b=40), height=600)
    return fig

# =============================================================================
# INTERFACE DO STREAMLIT
# =============================================================================
col1, col2 = st.columns([1, 2])
with col1:
    st.header("1. Adicionar Nova Peça")
    
    # AQUI: Simples e direto como no original
    num_vertices_total = st.number_input("Número de Vértices da Peça", min_value=3, value=8, step=1)
    
    with st.form(key="orthogonal_piece_form", clear_on_submit=True):
        piece_name = st.text_input("Nome da Peça", f"Peca {len(st.session_state.orthogonal_pieces) + 1}")
        quantity = st.number_input("Quantidade", min_value=1, value=1, step=1)
        
        # Recriar o DataFrame com zeros sempre que o script corre (garante tamanho certo)
        # Isto funciona porque quando mudamos o number_input, o script re-executa.
        df_rows = [{'X': 0.0, 'Y': 0.0} for _ in range(num_vertices_total)]
        df = pd.DataFrame(df_rows)
        
        st.write("Insira as coordenadas dos vértices da peça completa:")
        edited_df = st.data_editor(df, num_rows="fixed", column_config={"X": st.column_config.NumberColumn(format="%.1f"), "Y": st.column_config.NumberColumn(format="%.1f")})
        
        if st.form_submit_button("Adicionar Peça à Lista"):
            piece_data = {"name": piece_name, "quantity": quantity, "vertices": edited_df.to_dict('records'), "is_confirmed": False, "decomposition": {'mae': [], 'filhas': []}}
            st.session_state.orthogonal_pieces.append(piece_data)
            st.success(f"Peça '{piece_name}' adicionada!")

with col2:
    st.header("2. Decompor Peças Criadas")
    if not st.session_state.orthogonal_pieces:
        st.info("Nenhuma peça foi adicionada ainda.")
    else:
        for i, piece in enumerate(st.session_state.orthogonal_pieces):
            st.markdown("---")
            col_fig, col_tool = st.columns(2)
            with col_fig:
                st.write(f"**{piece['name']} (Qtd: {piece['quantity']})**")
                fig = plot_piece(piece); st.plotly_chart(fig, use_container_width=True)
            with col_tool:
                if piece['is_confirmed']:
                    st.success(f"Decomposição da Peça '{piece['name']}' confirmada!")
                    btn_cols = st.columns(2)
                    with btn_cols[0]:
                        if st.button("Refazer Decomposição", key=f"redo_{i}"):
                            piece['is_confirmed'] = False; piece['decomposition'] = {'mae': [], 'filhas': []}; st.rerun()
                    with btn_cols[1]:
                        if st.button("Remover Peça", key=f"remove_{i}", type="secondary"):
                            st.session_state.orthogonal_pieces.pop(i); st.rerun()
                else:
                    with st.expander("Ferramentas de Decomposição", expanded=True):
                        st.subheader("Opção Rápida (Peça Simples)")
                        if st.button("Confirmar como Peça Única (Só Mãe)", key=f"confirm_simple_{i}"):
                            piece['decomposition']['mae'] = piece['vertices']; piece['decomposition']['filhas'] = []; piece['is_confirmed'] = True; st.rerun()
                        st.markdown("---")
                        st.subheader("Opção Manual: Passo 1 (Mãe)")
                        
                        if not piece['decomposition']['mae']:
                            # Lógica para tabela da Mãe
                            num_mae_verts = st.number_input("Nº de Vértices da Mãe", min_value=3, value=4, step=1, key=f"num_mae_verts_{i}")
                            mae_df = pd.DataFrame([{'X': 0.0, 'Y': 0.0} for _ in range(num_mae_verts)])
                            edited_mae_df = st.data_editor(mae_df, num_rows="fixed", key=f"mae_editor_{i}", column_config={"X": st.column_config.NumberColumn(format="%.1f"), "Y": st.column_config.NumberColumn(format="%.1f")})
                            if st.button("Definir como Mãe", key=f"confirm_mae_{i}"):
                                piece['decomposition']['mae'] = edited_mae_df.to_dict('records'); st.rerun()
                        else:
                            st.success("Peça Mãe definida.")
                            
                        if piece['decomposition']['mae']:
                            st.subheader("Opção Manual: Passo 2 (Filhas)")
                            # Lógica para tabela das Filhas
                            num_filha_verts = st.number_input("Nº de Vértices da Próxima Filha", min_value=3, value=4, step=1, key=f"num_filha_verts_{i}")
                            filha_df = pd.DataFrame([{'X': 0.0, 'Y': 0.0} for _ in range(num_filha_verts)])
                            edited_filha_df = st.data_editor(filha_df, num_rows="fixed", key=f"filha_editor_{i}", column_config={"X": st.column_config.NumberColumn(format="%.1f"), "Y": st.column_config.NumberColumn(format="%.1f")})
                            if st.button(f"Adicionar Peça Filha {len(piece['decomposition']['filhas']) + 1}", key=f"add_filha_{i}"):
                                piece['decomposition']['filhas'].append(edited_filha_df.to_dict('records')); st.rerun()
                        
                        st.subheader("Opção Manual: Passo 3 (Finalizar)")
                        if st.button("Confirmar Decomposição Manual", key=f"confirm_manual_{i}", type="primary"):
                            if not piece['decomposition']['mae']:
                                st.error("É necessário definir a Peça Mãe antes de confirmar.")
                            else:
                                piece['is_confirmed'] = True; st.success("Decomposição confirmada!"); st.rerun()

st.markdown("---")
st.header("3. Iniciar Otimização")

H_strip = st.number_input("Altura Total da Tira (H)", min_value=1.0, value=20.0, step=0.5, help="Defina a altura fixa da sua tira de material.")
time_limit = st.number_input("Limite de Tempo (segundos)", value=60, min_value=5, max_value=600, step=5, key="time_ortho")

all_confirmed = all(p['is_confirmed'] for p in st.session_state.orthogonal_pieces) if st.session_state.orthogonal_pieces else False

if st.button("Otimizar Disposição das Peças", type="primary", disabled=(not all_confirmed)):
    with st.spinner("A calcular a solução..."):
        try:
            processed_data = preprocess_data_for_solver(st.session_state.orthogonal_pieces)
            model, status, all_rects_data = solve_orthogonal_packing(processed_data, H_strip, time_limit)
            
            if status == pl.LpStatusOptimal:
                st.success("Solução ótima encontrada!")
            elif status == pl.LpStatusInfeasible:
                st.error("Problema Infactível.")
            elif status == pl.LpStatusUndefined:
                st.warning("Solução não definida (limite de tempo).")
            
            if model.objective.value() is not None:
                W_final = model.objective.value()
                st.metric(label="Comprimento Mínimo da Tira (W)", value=f"{W_final:.2f}")
                solution_fig = plot_final_solution(model, H_strip, all_rects_data)
                st.plotly_chart(solution_fig, use_container_width=True)
            else:
                 st.error("Nenhuma solução foi encontrada.")
        except Exception as e:
            st.error(f"Erro: {e}"); st.exception(e)