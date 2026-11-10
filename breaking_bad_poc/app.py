import os
import hashlib
from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.breaking_bad.config import FEATURES, MODEL_PATH
from src.breaking_bad.ingestion import read_uploaded_file, classify_local_rows
from src.breaking_bad.training import load_trusted_model

st.set_page_config(page_title="Delivery Risk Intelligence", page_icon="📦", layout="wide")

# Optional shared-password gate for local/private deployments.
def authenticate():
    expected = os.getenv("APP_PASSWORD")
    if not expected:
        st.sidebar.info("Acesso local sem senha. Configure APP_PASSWORD para habilitar autenticação.")
        return True
    if st.session_state.get("authenticated"):
        return True
    st.title("🔐 Acesso restrito")
    pwd = st.text_input("Senha de acesso", type="password")
    if st.button("Entrar", type="primary"):
        if hashlib.sha256(pwd.encode()).digest() == hashlib.sha256(expected.encode()).digest():
            st.session_state.authenticated = True
            st.rerun()
        st.error("Credenciais inválidas.")
    return False

if not authenticate():
    st.stop()

model, metadata = load_trusted_model()
st.title("📦 Delivery Risk Intelligence")
st.caption("Breaking Bad · POC acadêmica de priorização de entregas")
st.warning("Esta aplicação é uma prova de conceito. O modelo Olist não representa desempenho validado no Mercado Livre atual.")

with st.sidebar:
    st.header("Navegação")
    page = st.radio("Área", ["Visão geral", "Importar pedidos", "Sobre o modelo", "Segurança e metodologia"])
    st.divider()
    st.caption("Execução local · Sem integração com ERP ou API do marketplace")

if page == "Visão geral":
    st.subheader("Visão operacional")
    if "orders_df" not in st.session_state:
        st.info("Importe um CSV ou XLSX na seção **Importar pedidos** para iniciar a análise.")
        a, b, c = st.columns(3)
        a.metric("Modelo", "Disponível" if model else "Não treinado")
        b.metric("Base local", "Não carregada")
        c.metric("Previsões", "—")
    else:
        x = st.session_state.orders_df
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pedidos importados", f"{len(x):,}")
        c2.metric("Elegíveis para previsão", f"{int(x.eligible_for_prediction.sum()):,}")
        c3.metric("Fora da previsão", f"{int((~x.eligible_for_prediction).sum()):,}")
        c4.metric("Modelo", "Ativo" if model else "Pendente")
        if "risk_score" in x:
            scored = x[x.risk_score.notna()].copy()
            if not scored.empty:
                left, right = st.columns(2)
                with left:
                    st.plotly_chart(px.histogram(scored, x="risk_score", nbins=20,
                        title="Distribuição das pontuações de risco", labels={"risk_score":"Pontuação (0–1)"}), use_container_width=True)
                with right:
                    regional = scored.groupby("customer_state", as_index=False).agg(
                        pedidos=("risk_score", "size"), risco_medio=("risk_score", "mean"))
                    st.plotly_chart(px.bar(regional, x="customer_state", y="pedidos",
                        color="risco_medio", title="Pedidos por UF e risco médio",
                        labels={"customer_state":"UF", "pedidos":"Pedidos", "risco_medio":"Risco médio"}), use_container_width=True)

elif page == "Importar pedidos":
    st.subheader("Importação e validação")
    st.write("Envie a exportação atualizada em CSV ou XLSX. O arquivo é processado em memória nesta sessão.")
    uploaded = st.file_uploader("Arquivo de pedidos", type=["csv", "xlsx"], help="Máximo de 10 MB e 50 mil linhas.")
    if uploaded is not None:
        try:
            raw_df = read_uploaded_file(uploaded)
            # Repeated order references are ambiguous; retain first occurrence and report duplicates.
            duplicate_count = int(raw_df["pedido_ref"].duplicated(keep="first").sum())
            raw_df = raw_df.drop_duplicates(subset=["pedido_ref"], keep="first").copy()
            classified = classify_local_rows(raw_df)
            model, metadata = load_trusted_model()
            if model is not None:
                eligible = classified.eligible_for_prediction
                if eligible.any():
                    model_input = classified.loc[eligible, FEATURES].copy()
                    for col in ["customer_state", "zip_prefix"]:
                        model_input[col] = model_input[col].astype(object).where(model_input[col].notna(), None)
                    classified.loc[eligible, "risk_score"] = model.predict_proba(model_input)[:, 1]
                    classified["risk_level"] = "Não pontuado"
                    classified.loc[eligible, "risk_level"] = pd.cut(
                        classified.loc[eligible, "risk_score"], bins=[-0.001, .33, .66, 1.001],
                        labels=["Baixo", "Moderado", "Alto"]).astype(str)
                else:
                    classified["risk_score"] = float("nan")
                    classified["risk_level"] = "Não pontuado"
            else:
                classified["risk_score"] = float("nan")
                classified["risk_level"] = "Modelo não treinado"
            st.session_state.orders_df = classified
            st.session_state.source_name = uploaded.name
            st.session_state.duplicate_count = duplicate_count
            st.success("Arquivo validado e processado.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Linhas recebidas", f"{len(raw_df) + duplicate_count:,}")
            m2.metric("Referências duplicadas", f"{duplicate_count:,}")
            m3.metric("Elegíveis", f"{int(classified.eligible_for_prediction.sum()):,}")
            if model is None:
                st.warning("Modelo não treinado. O sistema validou os pedidos, mas não gerou pontuações fictícias. Adicione os CSVs oficiais da Olist e execute `python train.py`.")
            else:
                st.success(f"Pontuação calculada com {metadata['model_name']} · versão treinada em {metadata['trained_at_utc']}.")
            st.markdown("#### Resultado da validação")
            reasons = classified.eligibility_reason.value_counts().rename_axis("Motivo").reset_index(name="Pedidos")
            st.dataframe(reasons, use_container_width=True, hide_index=True)
            display_cols = ["display_id", "data_compra", "data_prometida_original", "customer_state", "zip_prefix",
                            "item_count", "total_price", "total_freight", "risk_score", "risk_level", "eligibility_reason"]
            if model is not None:
                show = classified[classified.eligible_for_prediction].sort_values("risk_score", ascending=False)
            else:
                show = classified.copy()
            st.markdown("#### Pedidos e prioridades")
            st.dataframe(show[[c for c in display_cols if c in show.columns]], use_container_width=True, hide_index=True)
            # Export only minimized operational fields; do not export full CEP or raw account/shipment IDs.
            export_cols = ["display_id", "data_compra", "data_prometida_original", "customer_state", "zip_prefix",
                           "risk_score", "risk_level", "eligibility_reason"]
            export = show[[c for c in export_cols if c in show.columns]].to_csv(index=False).encode("utf-8-sig")
            st.download_button("Baixar lista minimizada (CSV)", export, "prioridades_entrega.csv", "text/csv")
        except Exception as exc:
            st.error(f"Não foi possível processar o arquivo: {exc}")

elif page == "Sobre o modelo":
    st.subheader("Modelo de Machine Learning")
    if metadata is None:
        st.warning("Ainda não existe modelo treinado. O treinamento requer os CSVs oficiais da Olist em `data/olist/`.")
        st.code("python train.py", language="bash")
        st.write("O pipeline compara Regressão Logística e Random Forest usando divisão temporal. A seleção ocorre na validação; o teste final permanece separado.")
    else:
        st.metric("Algoritmo selecionado", metadata["model_name"])
        st.write(f"Treinado em (UTC): {metadata['trained_at_utc']}")
        st.write(f"Registros treino + validação: {metadata['training_rows']:,} · Teste: {metadata['test_rows']:,}")
        st.markdown("#### Métricas no teste temporal")
        metrics = metadata["test_metrics"]
        cols = st.columns(5)
        for col, key, label in zip(cols, ["pr_auc", "precision", "recall", "f1", "top_20pct_lift"],
                                   ["PR-AUC", "Precisão", "Recall", "F1", "Lift top 20%"]):
            val = metrics.get(key)
            col.metric(label, "—" if val is None else f"{val:.3f}")
        st.json(metadata["validation_metrics"])
        st.caption(metadata["warning"])

else:
    st.subheader("Segurança, governança e metodologia")
    st.markdown("""
    **Controles implementados**
    - Upload restrito a CSV/XLSX, validação de assinatura, tamanho e estrutura ZIP do XLSX.
    - Limite de 10 MB, 50 mil linhas e 50 MB descompactados.
    - Leitura de valores de planilha sem executar fórmulas ou macros.
    - Modelo carregado exclusivamente do diretório local controlado; nenhum upload de modelo.
    - Identificadores excluídos das features; CEP completo reduzido a prefixo de cinco dígitos.
    - Nenhuma escrita no ERP, chamada à API do Mercado Livre ou envio de mensagens.
    - Previsões mantidas apenas na sessão e exportação minimizada.

    **Regras metodológicas**
    - O alvo Olist é entrega em dia de calendário posterior à promessa.
    - Entrega real, status final e resultado não entram nas features.
    - Pedidos locais só são pontuados se a promessa estiver marcada como confiável e capturada inicialmente.
    - Pedidos entregues, cancelados ou indisponíveis não são previsões preventivas.
    - Modalidade Flex/Agência/Full é contexto local, não feature aprendida artificialmente na Olist.

    **Limites conhecidos**
    - Olist representa 2016–2018 e não identifica Flex, Agência ou Full.
    - Compatibilidade de campos não comprova generalização para o Mercado Livre atual.
    - Pontuação é ranking de risco, não probabilidade calibrada, salvo validação específica de calibração.
    - A aplicação é local; não foi projetada para exposição pública.
    """)
