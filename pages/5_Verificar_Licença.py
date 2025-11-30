# pages/5_Verificar_Licença.py
import streamlit as st
import gurobipy as gp
import sys # Para verificar a versão

st.set_page_config(page_title="Verificar Licença Gurobi", layout="wide")
st.title("Diagnóstico da Licença Gurobi")

st.info("""
Esta página apenas tem o intuito de testar se a licença do Gurobi está a funcionar, testando a ligação WLS usando o método `gp.Env(params=...)`.
Só funcionará DEPOIS de executar `python -m pip install --upgrade gurobipy`
no terminal correto.
""")

# --- Credenciais WLS (do seu ficheiro gurobi.lic) ---
WLS_ACCESS_ID = "b04e9716-989f-4118-b69c-32974938514e"
WLS_SECRET = "c369d181-29b4-4b26-b3e1-9c2f480fdb32"
# Gurobi ID numérico
LICENSE_ID = 2737454 
# ---

st.write(f"**Versão do Gurobipy a ser usada:** `{gp.gurobi.version()}`")
st.write(f"**Versão do Python:** `{sys.version.split(' ')[0]}`")

if st.button("Executar Teste de Licença Gurobi (WLS)", type="primary"):
    
    st.subheader("Resultado do Teste:")
    
    try:
        # --- PASSO 1: Definir parâmetros (Baseado no Gurobi AI, Step 3, Method 2) ---
        st.write(f"1. A definir parâmetros WLS (LicenseID: {LICENSE_ID})...")
        wls_options = {
            "WLSACCESSID": WLS_ACCESS_ID,
            "WLSSECRET": WLS_SECRET,
            "LICENSEID": LICENSE_ID, # Passado como inteiro
        }

        # --- PASSO 2: Criar um ambiente Gurobi ---
        st.write("2. A tentar criar um Ambiente Gurobi (gp.Env)...")
        with st.spinner("A carregar licença WLS..."):
            # Este é o método MODERNO (requer gurobipy 10+)
            # Isto falhou antes porque o seu gurobipy estava desatualizado
            env = gp.Env(params=wls_options)
            
        st.success("2. Ambiente Gurobi WLS inicializado com SUCESSO.")

        # --- PASSO 3: Testar o limite da licença ---
        st.write("3. A tentar criar um modelo com 3000 variáveis (para testar o limite)...")
        with st.spinner("A construir modelo de teste..."):
            m = gp.Model("teste_licenca", env=env)
            x = m.addVars(3000, name="x") 
            m.update()
        
        st.success("3. Modelo com 3000 variáveis criado com SUCESSO.")
        
        st.balloons()
        st.success("A sua licença WLS ilimitada foi lida corretamente!")
        st.info("O problema original na Página 4 deve estar resolvido.")

    except gp.GurobiError as e:
        st.error("O Gurobi devolveu um erro.")
        st.error(f"Detalhe do Erro: {e}")
        st.warning("""
        Causas comuns:
        1. O 'upgrade' do gurobipy falhou.
        2. As credenciais WLS estão incorretas (improvável).
        """)
    
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado do Python: {e}")
        st.exception(e)