# pages/6_Modo_Separado.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
import numpy as np
import pulp as pl

# --- Configuração da Página ---
st.set_page_config(page_title="Modo Separado", layout="wide")
st.title("Modo Separado: Construção de Peças Compostas")

st.info("""
**Instruções:**
1. Defina os vértices da **Parte 1** e da **Parte 2** (podem ser formas em 'L', 'T', etc.).
2. O sistema irá **decompor imediatamente** essas formas em retângulos simples.
3. A peça final será guardada como um conjunto de retângulos (Mãe + Filhas) com posições fixas.
""")

# --- GESTÃO DE ESTADO ---
if 'p6_df_part1' not in st.session_state:
    st.session_state.p6_df_part1 = pd.DataFrame([{'X': 0.0, 'Y': 0.0}, {'X': 2.0, 'Y': 0.0}, {'X': 2.0, 'Y': 4.0}, {'X': 0.0, 'Y': 4.0}])

if 'p6_df_part2' not in st.session_state:
    st.session_state.p6_df_part2 = pd.DataFrame([{'X': 2.0, 'Y': 0.0}, {'X': 4.0, 'Y': 0.0}, {'X': 4.0, 'Y': 2.0}, {'X': 2.0, 'Y': 2.0}])

if 'orthogonal_pieces' not in st.session_state:
    st.session_state.orthogonal_pieces = []

# =============================================================================
# FUNÇÕES GEOMÉTRICAS E DE DECOMPOSIÇÃO
# =============================================================================
def force_adjust_rows(df, target_rows):
    """Ajusta dinamicamente o número de linhas da tabela."""
    current_rows = len(df)
    if current_rows == target_rows: return df
    if current_rows < target_rows:
        new_data = pd.DataFrame([{'X': 0.0, 'Y': 0.0} for _ in range(target_rows - current_rows)])
        return pd.concat([df, new_data], ignore_index=True)
    else:
        return df.iloc[:target_rows]

def auto_decompose_polygon(vertices):
    """Parte um polígono complexo em lista de dicionários {'x','y','w','h'}."""
    rects = []
    # Algoritmo de varrimento vertical simples para polígonos ortogonais
    y_levels = sorted(list(set(v['Y'] for v in vertices)))
    for i in range(len(y_levels) - 1):
        y_min, y_max = y_levels[i], y_levels[i+1]
        vertical_edges = []
        for j in range(len(vertices)):
            v1 = vertices[j]; v2 = vertices[(j + 1) % len(vertices)]
            # Encontrar arestas verticais que cruzam esta faixa Y
            if abs(v1['X'] - v2['X']) < 1e-9: # X igual (vertical)
                min_v_y, max_v_y = min(v1['Y'], v2['Y']), max(v1['Y'], v2['Y'])
                if min_v_y <= y_min and max_v_y >= y_max:
                    vertical_edges.append(v1['X'])
        vertical_edges.sort()
        # Criar retângulos entre pares de arestas
        for j in range(0, len(vertical_edges), 2):
            if j+1 < len(vertical_edges):
                x_min, x_max = vertical_edges[j], vertical_edges[j+1]
                if x_max > x_min:
                    rects.append({'x': x_min, 'y': y_min, 'w': x_max - x_min, 'h': y_max - y_min})
    return rects

def rect_dict_to_verts(r):
    """Converte formato {x,y,w,h} para lista de vértices para plot/armazenamento."""
    return [
        {'X': r['x'], 'Y': r['y']},
        {'X': r['x'] + r['w'], 'Y': r['y']},
        {'X': r['x'] + r['w'], 'Y': r['y'] + r['h']},
        {'X': r['x'], 'Y': r['y'] + r['h']}
    ]

def get_rect_props(r):
    """Calcula propriedades essenciais para o solver."""
    center_x = r['x'] + r['w'] / 2
    center_y = r['y'] + r['h'] / 2
    return {'w': r['w'], 'h': r['h'], 'center_x': center_x, 'center_y': center_y}

# =============================================================================
# LÓGICA DE CRIAÇÃO DA PEÇA (O Cérebro da Página)
# =============================================================================
def create_and_decompose_piece(name, qty, verts1, verts2, mother_choice):
    """
    1. Decompõe a Parte 1 e a Parte 2 em retângulos simples.
    2. Identifica a 'Mãe Real' (o maior retângulo da parte escolhida).
    3. Calcula Dx e Dy para TODOS os outros retângulos.
    4. Devolve a estrutura pronta para o solver.
    """
    # 1. Decompor ambas as partes
    rects1 = auto_decompose_polygon(verts1)
    rects2 = auto_decompose_polygon(verts2)
    
    if not rects1 or not rects2:
        return None # Erro na geometria

    # 2. Identificar quem é a base (Mãe)
    if mother_choice == "Parte 1":
        # A mãe real é o maior retângulo da Parte 1
        rects1.sort(key=lambda r: r['w']*r['h'], reverse=True)
        real_mae_dict = rects1[0]
        others = rects1[1:] + rects2 # As 'irmãs' da parte 1 + todas da parte 2
    else:
        # A mãe real é o maior retângulo da Parte 2
        rects2.sort(key=lambda r: r['w']*r['h'], reverse=True)
        real_mae_dict = rects2[0]
        others = rects2[1:] + rects1

    # 3. Calcular propriedades base
    mae_props = get_rect_props(real_mae_dict)
    
    # Preparar estrutura de armazenamento
    # 'mae' guarda os vértices do retângulo principal
    # 'filhas' guarda os vértices das outras partes
    # IMPORTANTE: O Solver vai precisar de w_orig, h_orig, Dx, Dy.
    # Vamos guardar isso dentro da estrutura 'filhas' para facilitar.
    
    mae_verts_final = rect_dict_to_verts(real_mae_dict)
    filhas_final_list = []
    
    for rect in others:
        f_props = get_rect_props(rect)
        D_x = f_props['center_x'] - mae_props['center_x']
        D_y = f_props['center_y'] - mae_props['center_y']
        
        # Guardamos os vértices (para visualização) E os metadados (para o solver)
        f_verts = rect_dict_to_verts(rect)
        # Injetamos metadados no primeiro vértice para o preprocessador ler depois,
        # ou melhor, confiamos que o preprocessador recalcula se os vértices estiverem certos.
        # Para garantir, vamos guardar a peça JÁ preprocessada na lista global.
        
        filhas_final_list.append(f_verts)

    return {
        "name": name,
        "quantity": qty,
        "vertices": verts1 + verts2, # Apenas para referência do contorno bruto
        "is_confirmed": True,
        "decomposition": {
            'mae': mae_verts_final, 
            'filhas': filhas_final_list
        }
    }

# =============================================================================
# PREPROCESSADOR PARA O SOLVER
# =============================================================================
def preprocess_data_for_solver(st_pieces):
    """Converte a estrutura visual (vértices) em estrutura matemática (Dx, Dy)."""
    pieces_data = []
    piece_counter = 1 
    
    for st_piece in st_pieces:
        for _ in range(st_piece['quantity']):
            # Recalcular propriedades da Mãe
            # Nota: st_piece['decomposition']['mae'] já é um retângulo simples agora!
            mae_verts = st_piece['decomposition']['mae']
            mae_props = calculate_rect_properties_simple(mae_verts)
            
            filhas_data = []
            for f_verts in st_piece['decomposition']['filhas']:
                f_props = calculate_rect_properties_simple(f_verts)
                D_x = f_props['center_x'] - mae_props['center_x']
                D_y = f_props['center_y'] - mae_props['center_y']
                filhas_data.append({
                    'w_orig': f_props['w'], 
                    'h_orig': f_props['h'], 
                    'Dx': D_x, 
                    'Dy': D_y
                })
            
            pieces_data.append({
                'id': piece_counter,
                'name': st_piece['name'],
                'mae': {'w_orig': mae_props['w'], 'h_orig': mae_props['h']},
                'filhas': filhas_data
            })
            piece_counter += 1
    return pieces_data

def calculate_rect_properties_simple(vertices):
    x_c = [v['X'] for v in vertices]; y_c = [v['Y'] for v in vertices]
    w = max(x_c) - min(x_c); h = max(y_c) - min(y_c)
    return {'w': w, 'h': h, 'center_x': min(x_c) + w/2, 'center_y': min(y_c) + h/2}

# =============================================================================
# VISUALIZAÇÃO
# =============================================================================
def plot_preview_input(df1, df2):
    fig = go.Figure()
    x1 = df1['X'].tolist() + [df1['X'].iloc[0]]; y1 = df1['Y'].tolist() + [df1['Y'].iloc[0]]
    fig.add_trace(go.Scatter(x=x1, y=y1, fill='toself', name='Parte 1 (Input)', line=dict(color='blue', dash='dash')))
    x2 = df2['X'].tolist() + [df2['X'].iloc[0]]; y2 = df2['Y'].tolist() + [df2['Y'].iloc[0]]
    fig.add_trace(go.Scatter(x=x2, y=y2, fill='toself', name='Parte 2 (Input)', line=dict(color='green', dash='dash')))
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10), title="Input Bruto")
    return fig

def plot_decomposed_preview(piece):
    fig = go.Figure()
    decomp = piece['decomposition']
    # Mãe
    mx = [v['X'] for v in decomp['mae']] + [decomp['mae'][0]['X']]
    my = [v['Y'] for v in decomp['mae']] + [decomp['mae'][0]['Y']]
    fig.add_trace(go.Scatter(x=mx, y=my, fill='toself', fillcolor='rgba(0,0,255,0.6)', line=dict(color='black'), name='Mãe Real'))
    # Filhas
    for i, f_verts in enumerate(decomp['filhas']):
        fx = [v['X'] for v in f_verts] + [f_verts[0]['X']]
        fy = [v['Y'] for v in f_verts] + [f_verts[0]['Y']]
        fig.add_trace(go.Scatter(x=fx, y=fy, fill='toself', fillcolor='rgba(0,255,0,0.6)', line=dict(color='black'), name=f'Filha {i+1}'))
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10), title=f"Peça Decomposta: {piece['name']}")
    return fig

def plot_final_solution(model, H, all_rects_data):
    fig = go.Figure()
    W_sol = model.objective.value()
    # Cores por ID da peça original
    unique_piece_ids = sorted(list(set(r['piece_id'] for r in all_rects_data.values())))
    colors = plotly.colors.qualitative.Plotly
    color_map = {pid: colors[i % len(colors)] for i, pid in enumerate(unique_piece_ids)}
    
    for rect_id, data in all_rects_data.items():
        xc = model.variablesDict()[f'Xc_{rect_id}'].varValue
        yc = model.variablesDict()[f'Yc_{rect_id}'].varValue
        wf = model.variablesDict()[f'Wf_{rect_id}'].varValue
        hf = model.variablesDict()[f'Hf_{rect_id}'].varValue
        
        x0 = xc - 0.5 * wf; y0 = yc - 0.5 * hf
        
        # Desenhar Retângulo
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x0+wf, y1=y0+hf, 
                      line=dict(color="Black", width=1), fillcolor=color_map[data['piece_id']], opacity=0.8)
        # Texto
        fig.add_annotation(x=xc, y=yc, text=f"P{data['piece_id']}", showarrow=False, font=dict(color="white", size=10))

    fig.add_shape(type="rect", x0=0, y0=0, x1=W_sol, y1=H, line=dict(color="Black", width=3), layer="below")
    fig.update_layout(title=f"Solução Final (W = {W_sol:.2f})", xaxis=dict(range=[-1, W_sol+2], showgrid=True), yaxis=dict(range=[-1, H+1], showgrid=True, scaleanchor="x", scaleratio=1))
    return fig

# =============================================================================
# SOLVER (Mesmo da Pág 3, sem alterações de lógica)
# =============================================================================
def solve_orthogonal_packing(pieces_data, H, time_limit):
    model = pl.LpProblem("Orthogonal_Item_Packing", pl.LpMinimize)
    piece_ids = [p['id'] for p in pieces_data]; all_rects = {} 
    rect_counter = 1
    for p in pieces_data:
        p['mae']['rect_id'] = f"r{rect_counter}"; p['mae']['piece_id'] = p['id']; all_rects[f"r{rect_counter}"] = p['mae']; rect_counter += 1
        for f in p['filhas']:
            f['rect_id'] = f"r{rect_counter}"; f['piece_id'] = p['id']; all_rects[f"r{rect_counter}"] = f; rect_counter += 1
    
    # Big M
    M = sum(r['w_orig'] for r in all_rects.values()) + sum(r['h_orig'] for r in all_rects.values()) + H + 1000
    W = pl.LpVariable("W", lowBound=0.0)
    
    Xc_mae = pl.LpVariable.dicts("Xc_mae", piece_ids, lowBound=0.0); Yc_mae = pl.LpVariable.dicts("Yc_mae", piece_ids, lowBound=0.0)
    z = pl.LpVariable.dicts("z", [(k, m) for k in piece_ids for m in range(1, 5)], cat="Binary")
    Wf = pl.LpVariable.dicts("Wf", all_rects.keys(), lowBound=0.0); Hf = pl.LpVariable.dicts("Hf", all_rects.keys(), lowBound=0.0)
    Xc = pl.LpVariable.dicts("Xc", all_rects.keys(), lowBound=0.0); Yc = pl.LpVariable.dicts("Yc", all_rects.keys(), lowBound=0.0)
    
    rect_pairs = []
    keys = list(all_rects.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            id1, id2 = keys[i], keys[j]
            if all_rects[id1]['piece_id'] != all_rects[id2]['piece_id']:
                rect_pairs.append((id1, id2))
    
    R = pl.LpVariable.dicts("R", rect_pairs + [(b,a) for a,b in rect_pairs], cat="Binary")
    U = pl.LpVariable.dicts("U", rect_pairs + [(b,a) for a,b in rect_pairs], cat="Binary")

    model += W, "Min_Width"
    
    for p in pieces_data:
        k = p['id']; mae = p['mae']; mae_id = mae['rect_id']
        model += pl.lpSum(z[k, m] for m in range(1, 5)) == 1
        model += Wf[mae_id] == (z[k,1] + z[k,3]) * mae['w_orig'] + (z[k,2] + z[k,4]) * mae['h_orig']
        model += Hf[mae_id] == (z[k,1] + z[k,3]) * mae['h_orig'] + (z[k,2] + z[k,4]) * mae['w_orig']
        model += Xc[mae_id] == Xc_mae[k]; model += Yc[mae_id] == Yc_mae[k]
        
        for f in p['filhas']:
            f_id = f['rect_id']
            model += Wf[f_id] == (z[k,1] + z[k,3]) * f['w_orig'] + (z[k,2] + z[k,4]) * f['h_orig']
            model += Hf[f_id] == (z[k,1] + z[k,3]) * f['h_orig'] + (z[k,2] + z[k,4]) * f['w_orig']
            # Lógica Delta atualizada
            delta_x = z[k,1]*f['Dx'] + z[k,2]*f['Dy'] - z[k,3]*f['Dx'] - z[k,4]*f['Dy']
            delta_y = z[k,1]*f['Dy'] - z[k,2]*f['Dx'] - z[k,3]*f['Dy'] + z[k,4]*f['Dx']
            model += Xc[f_id] == Xc[mae_id] + delta_x
            model += Yc[f_id] == Yc[mae_id] + delta_y

    for r_id in all_rects:
        model += Xc[r_id] + 0.5*Wf[r_id] <= W
        model += Yc[r_id] + 0.5*Hf[r_id] <= H
        model += Xc[r_id] - 0.5*Wf[r_id] >= 0
        model += Yc[r_id] - 0.5*Hf[r_id] >= 0

    for id1, id2 in rect_pairs:
        model += Xc[id1] + 0.5*Wf[id1] <= Xc[id2] - 0.5*Wf[id2] + M*R[(id1,id2)]
        model += Xc[id2] + 0.5*Wf[id2] <= Xc[id1] - 0.5*Wf[id1] + M*R[(id2,id1)]
        model += Yc[id1] + 0.5*Hf[id1] <= Yc[id2] - 0.5*Hf[id2] + M*U[(id1,id2)]
        model += Yc[id2] + 0.5*Hf[id2] <= Yc[id1] - 0.5*Hf[id1] + M*U[(id2,id1)]
        model += R[(id1,id2)] + R[(id2,id1)] + U[(id1,id2)] + U[(id2,id1)] <= 3

    status = model.solve(pl.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    return model, status, all_rects

# =============================================================================
# INTERFACE
# =============================================================================
c_input, c_view = st.columns([1, 1.2])

with c_input:
    st.subheader("1. Definição Geométrica")
    piece_name = st.text_input("Nome da Peça", f"Peca_{len(st.session_state.orthogonal_pieces)+1}")
    qty = st.number_input("Quantidade", min_value=1, value=1)
    
    st.divider()
    st.write("**Parte 1 (Vértices)**")
    n1 = st.number_input("Nº Vértices P1", 3, 8, 4, key="n1")
    st.session_state.p6_df_part1 = force_adjust_rows(st.session_state.p6_df_part1, n1)
    ed1 = st.data_editor(st.session_state.p6_df_part1, key="ed1", hide_index=True)
    st.session_state.p6_df_part1 = ed1
    
    st.divider()
    st.write("**Parte 2 (Vértices)**")
    n2 = st.number_input("Nº Vértices P2", 3, 8, 4, key="n2")
    st.session_state.p6_df_part2 = force_adjust_rows(st.session_state.p6_df_part2, n2)
    ed2 = st.data_editor(st.session_state.p6_df_part2, key="ed2", hide_index=True)
    st.session_state.p6_df_part2 = ed2
    
    st.divider()
    mother_choice = st.radio("Parte Principal (Mãe)", ["Parte 1", "Parte 2"], horizontal=True)
    
    if st.button("✅ Decompor e Criar Peça", type="primary", use_container_width=True):
        v1 = ed1.to_dict('records')
        v2 = ed2.to_dict('records')
        
        # --- AQUI ESTÁ A MÁGICA: Decomposição IMEDIATA ---
        new_piece_struct = create_and_decompose_piece(piece_name, qty, v1, v2, mother_choice)
        
        if new_piece_struct:
            st.session_state.orthogonal_pieces.append(new_piece_struct)
            st.success("Peça decomposta e guardada com sucesso!")
        else:
            st.error("Erro na geometria dos polígonos.")

with c_view:
    st.subheader("Pré-visualização (Input)")
    st.plotly_chart(plot_preview_input(st.session_state.p6_df_part1, st.session_state.p6_df_part2), use_container_width=True)
    
    st.markdown("---")
    st.write("### Peças Prontas (Já Decompostas)")
    if st.session_state.orthogonal_pieces:
        for i, p in enumerate(st.session_state.orthogonal_pieces):
            with st.expander(f"{p['name']} (Qtd: {p['quantity']})", expanded=True):
                # Mostrar a decomposição real que vai para o solver
                fig_decomp = plot_decomposed_preview(p)
                st.plotly_chart(fig_decomp, use_container_width=True)
                if st.button("Remover", key=f"rm_{i}"):
                    st.session_state.orthogonal_pieces.pop(i); st.rerun()
    else:
        st.info("Crie peças para ver a sua decomposição aqui.")

st.markdown("---")
st.header("3. Otimizar")
H_strip = st.number_input("Altura da Tira (H)", 10.0, 100.0, 20.0)
t_lim = st.number_input("Tempo Limite (s)", 5, 600, 60)

if st.button("🚀 Otimizar Disposição", type="primary", disabled=not st.session_state.orthogonal_pieces):
    with st.spinner("A resolver..."):
        try:
            # Dados já estão decompostos, o preprocessador agora é simples
            data = preprocess_data_for_solver(st.session_state.orthogonal_pieces)
            model, status, res = solve_orthogonal_packing(data, H_strip, t_lim)
            
            if model.objective.value():
                st.success("Solução encontrada!")
                st.metric("W Final", f"{model.objective.value():.2f}")
                st.plotly_chart(plot_final_solution(model, H_strip, res), use_container_width=True)
            else:
                st.error("Sem solução encontrada.")
        except Exception as e:
            st.error(f"Erro: {e}")