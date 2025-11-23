# Página_Inicial.py
import streamlit as st

# Configuração da página (deve ser o primeiro comando Streamlit)
st.set_page_config(
    page_title="Otimização de Corte e Empacotamento",
    page_icon="📐",
    layout="wide"
)

# Título e descrição da página principal
st.title("Bem-vindo à Ferramenta de Otimização de Corte")
st.subheader("Projeto da UC de Modelos de Otimização e de Apoio à Decisão (MOADC)")

st.markdown("""
Esta aplicação foi desenvolvida no âmbito do Mestrado em Engenharia e Gestão da Cadeia de Abastecimento do ISEP.

O objetivo é aplicar modelos de programação linear e inteira para resolver problemas complexos de corte e empacotamento, minimizando o desperdício de material.

**Utilize o menu de navegação na barra lateral para selecionar o problema que deseja resolver.**
""")

st.info("Selecione uma das páginas na barra lateral para começar.", icon="👈")

# Apresentar as diferentes páginas com links diretos
st.subheader("Navegação Rápida")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("##### 1. Problema de Retângulos (SPP)")
        st.write("Otimiza a disposição de múltiplos retângulos numa tira de material, permitindo rotações de 90º para minimizar o comprimento total utilizado.")
        
        # O caminho do ficheiro foi atualizado para o novo nome
        st.page_link("pages/1_Retangulos.py", label="Aceder ao Otimizador de Retângulos", icon="📏")

with col2:
    with st.container(border=True):
        st.markdown("##### 2. Figuras Geométricas Complexas")
        st.write("Uma extensão do problema para lidar com figuras irregulares mais complexas")
        
        # --- ALTERAÇÃO ---
        # Removido "disabled=True" e alterado o label
        st.page_link("pages/2_Figuras_irregulares.py", label="Aceder ao Otimizador Geométrico", icon="💠")

with col3:
    with st.container(border=True):
        st.markdown("##### 3. Problema de Itens Ortogonais")
        st.write("Modela e otimiza o corte de polígonos ortogonais (em forma de 'L', 'T', 'cruz', etc.), permitindo rotações de 0º, 90º, 180º e 270º.")
        
        # --- ALTERAÇÃO ---
        # Removido "disabled=True" e alterado o label
        st.page_link("pages/3_Itens_Ortogonais.py", label="Aceder ao Otimizador Ortogonal", icon="➕")