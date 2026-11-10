import os
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.breaking_bad.config import FEATURES
from src.breaking_bad.ingestion import read_uploaded_file, classify_local_rows
from src.breaking_bad.training import load_trusted_model

st.set_page_config(page_title="Breaking Bad | Delivery Intelligence", page_icon="📦", layout="wide", initial_sidebar_state="expanded")

# ----------------------------- Visual system -----------------------------
NAVY = "#101B2D"
PANEL = "#17263B"
CYAN = "#42D6C7"
BLUE = "#5B8DEF"
ORANGE = "#FFB454"
RED = "#F16B6B"
GREEN = "#4CC9A6"
MUTED = "#9AAAC0"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {{ --bb-bg:{NAVY}; --bb-panel:{PANEL}; --bb-cyan:{CYAN}; }}
.stApp {{ background: radial-gradient(circle at 82% 0%, #1b3550 0, {NAVY} 42%); color:#F4F7FB; font-family:'DM Sans',sans-serif; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"] {{ background:#0B1423; border-right:1px solid #26374D; }}
[data-testid="stSidebar"] * {{ color:#EAF1FA; }}
.block-container {{ padding-top:1.6rem; padding-bottom:2.5rem; max-width:1500px; }}
h1,h2,h3 {{ font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; }}
.eyebrow {{ color:{CYAN}; font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }}
.hero-title {{ font-family:'Space Grotesk',sans-serif; font-size:2.25rem; font-weight:700; letter-spacing:-.06em; line-height:1.08; margin:.3rem 0 .35rem; }}
.hero-sub {{ color:#A9B8CC; font-size:.94rem; }}
.kpi {{ background:linear-gradient(145deg,#1A2B42,#142237); border:1px solid #2A405B; border-radius:16px; padding:17px 18px 15px; min-height:118px; box-shadow:0 8px 22px rgba(0,0,0,.12); }}
.kpi-label {{ color:#A9B8CC; font-size:.76rem; font-weight:600; }}
.kpi-value {{ color:#F7FAFF; font-family:'Space Grotesk',sans-serif; font-size:1.85rem; font-weight:700; margin-top:7px; line-height:1.15; }}
.kpi-note {{ color:{CYAN}; font-size:.7rem; margin-top:7px; }}
.section-head {{ display:flex; align-items:center; gap:9px; margin:22px 0 10px; font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:700; }}
.section-head:before {{ content:''; display:block; width:4px; height:19px; border-radius:5px; background:{CYAN}; }}
.insight {{ background:#182A40; border:1px solid #2B425E; border-left:4px solid {CYAN}; border-radius:12px; padding:13px 15px; color:#DCE7F5; font-size:.88rem; }}
.warning {{ background:#352C20; border:1px solid #6B5433; border-left:4px solid {ORANGE}; border-radius:12px; padding:13px 15px; color:#FCE7C7; font-size:.85rem; }}
div[data-testid="stMetric"] {{ background:#182A40; border:1px solid #2B425E; border-radius:14px; padding:12px 14px; }}
div[data-testid="stMetricLabel"] p {{ color:#A9B8CC !important; }}
div[data-testid="stMetricValue"] {{ color:#F7FAFF !important; }}
.stButton>button, .stDownloadButton>button {{ border-radius:10px; border:1px solid #35516F; background:#1A2D45; color:#F4F7FB; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ border-color:{CYAN}; color:{CYAN}; }}
[data-testid="stDataFrame"] {{ border:1px solid #2A405B; border-radius:12px; overflow:hidden; }}
hr {{ border-color:#2A405B; }}
small {{ color:#9AAAC0; }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#DCE7F5", size=11),
    margin=dict(l=12, r=12, t=42, b=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#B9C8DA")),
    xaxis=dict(gridcolor="#263A52", zerolinecolor="#263A52", linecolor="#263A52"),
    yaxis=dict(gridcolor="#263A52", zerolinecolor="#263A52", linecolor="#263A52"),
)


def apply_layout(fig, height=310):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    return fig


def money(v):
    return f"R$ {v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v):
    return "—" if pd.isna(v) else f"{v:.1%}"


def card(label, value, note="", accent=CYAN):
    st.markdown(f'''<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note" style="color:{accent}">{note}</div></div>''', unsafe_allow_html=True)


def parse_dates(df):
    date_cols = ["data_compra", "data_prometida_original", "Limite de despacho", "Despacho real", "Primeira tentativa", "Entrega real", "Ultima coleta"]
    for c in date_cols:
        if c in df:
            s = df[c].astype("string").replace({"-": pd.NA, "": pd.NA, "nan": pd.NA, "NaT": pd.NA})
            df[c + "_dt"] = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce")
    for c in ["Dias de atraso", "Atraso operacional (min)", "quantidade_itens", "valor_produtos", "valor_frete"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["UF"] = df.get("uf_destino", df.get("UF entrega", pd.Series(index=df.index, dtype="object"))).astype("string").str.upper().str.strip()
    df["Modalidade_norm"] = df["Modalidade"].astype("string").str.strip().str.title() if "Modalidade" in df else "Não informado"
    df["Status_norm"] = df["Status do envio"].astype("string").str.lower().str.strip() if "Status do envio" in df else "não informado"
    df["Resultado_norm"] = df["Resultado"].astype("string").str.strip() if "Resultado" in df else "Não informado"
    df["promessa_confiavel"] = (
        df.get("Promessa confiavel", pd.Series("", index=df.index)).astype("string").str.lower().eq("confiavel")
        & df.get("Origem da promessa", pd.Series("", index=df.index)).astype("string").str.lower().eq("captura_inicial")
    )
    df["mes_compra"] = df.get("data_compra_dt", pd.Series(pd.NaT, index=df.index)).dt.to_period("M").astype("string")
    df["atraso_confirmado"] = df["Resultado_norm"].isin(["Entregue atrasado", "Em transito atrasado", "Nao entregue"])
    df["entregue_no_prazo"] = df["Resultado_norm"].eq("Entregue no prazo")
    df["entregue_atrasado"] = df["Resultado_norm"].eq("Entregue atrasado")
    return df


def load_default_data():
    path = ROOT / "data" / "base_ecommerce_demo.xlsx"
    if path.exists():
        raw = pd.read_excel(path, sheet_name="Entregas", engine="openpyxl")
        raw.columns = [str(c).strip() for c in raw.columns]
        return raw, "base_ecommerce_demo.xlsx"
    return None, None


def enrich(raw):
    classified = classify_local_rows(raw.copy())
    return parse_dates(classified)


def authenticate():
    expected = os.getenv("APP_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("authenticated"):
        return True
    st.markdown('<div class="eyebrow">Área restrita</div><div class="hero-title">Acesso ao painel</div>', unsafe_allow_html=True)
    pwd = st.text_input("Senha", type="password")
    if st.button("Entrar", type="primary"):
        if hashlib.sha256(pwd.encode()).digest() == hashlib.sha256(expected.encode()).digest():
            st.session_state.authenticated = True
            st.rerun()
        st.error("Senha incorreta.")
    return False

if not authenticate():
    st.stop()

# ----------------------------- Sidebar -----------------------------
raw_default, default_name = load_default_data()
if "dashboard_raw" not in st.session_state:
    st.session_state.dashboard_raw = raw_default
    st.session_state.dashboard_source = default_name or "Nenhuma base carregada"

with st.sidebar:
    st.markdown(f'<div class="eyebrow">BREAKING BAD</div><div style="font-family:Space Grotesk;font-size:1.35rem;font-weight:700">Delivery<br><span style="color:{CYAN}">Intelligence</span></div>', unsafe_allow_html=True)
    st.caption("POC · Analytics & Risk Monitoring")
    st.divider()
    page = st.radio("NAVEGAÇÃO", ["01 · Visão executiva", "02 · Diagnóstico operacional", "03 · Delivery Risk", "04 · Modelo & qualidade"], label_visibility="visible")
    st.divider()
    st.markdown("**BASE DE DADOS**")
    uploaded = st.file_uploader("Atualizar base (XLSX ou CSV)", type=["xlsx", "csv"], help="O arquivo é processado nesta sessão. Limite de 10 MB e 50 mil linhas.")
    if uploaded is not None:
        try:
            raw = read_uploaded_file(uploaded)
            st.session_state.dashboard_raw = raw
            st.session_state.dashboard_source = uploaded.name
            st.success("Base atualizada nesta sessão.")
        except Exception as e:
            st.error(f"Falha na importação: {e}")
    st.caption(f"Fonte: {st.session_state.dashboard_source}")
    if st.button("Restaurar base de demonstração", use_container_width=True):
        st.session_state.dashboard_raw = raw_default
        st.session_state.dashboard_source = default_name or "Nenhuma base carregada"
        st.rerun()
    st.divider()
    st.caption("Execução local · Sem integração com ERP ou marketplace")

raw = st.session_state.dashboard_raw
if raw is None:
    st.markdown('<div class="hero-title">Sua operação, sob uma nova perspectiva.</div>', unsafe_allow_html=True)
    st.info("Carregue uma planilha XLSX ou CSV na barra lateral para começar.")
    st.stop()

df_all = enrich(raw)

# Global filters
with st.sidebar:
    st.markdown("**FILTROS**")
    uf_options = sorted(df_all["UF"].dropna().astype(str).unique().tolist())
    modality_options = sorted(df_all["Modalidade_norm"].dropna().astype(str).unique().tolist())
    selected_ufs = st.multiselect("UF de destino", uf_options, default=uf_options)
    selected_modalities = st.multiselect("Modalidade", modality_options, default=modality_options)
    if "data_compra_dt" in df_all and df_all["data_compra_dt"].notna().any():
        min_d = df_all["data_compra_dt"].min().date()
        max_d = df_all["data_compra_dt"].max().date()
        date_range = st.date_input("Período de compra", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    else:
        date_range = None

filtered = df_all[df_all["UF"].isin(selected_ufs) & df_all["Modalidade_norm"].isin(selected_modalities)].copy()
if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = filtered[filtered["data_compra_dt"].between(start, end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]

model, metadata = load_trusted_model()
trusted = filtered[filtered["promessa_confiavel"]].copy()
delivered_trusted = trusted[trusted["Status_norm"].eq("delivered")].copy()
open_eligible = filtered[filtered["eligible_for_prediction"]].copy()
late_confirmed = trusted[trusted["atraso_confirmado"]].copy()

# ----------------------------- Header -----------------------------
page_titles = {
    "01 · Visão executiva": ("COMMAND CENTER", "Visão executiva", "Uma leitura rápida da saúde operacional das entregas."),
    "02 · Diagnóstico operacional": ("OPERATIONAL ANALYTICS", "Diagnóstico operacional", "Encontre gargalos por região, modalidade e etapa logística."),
    "03 · Delivery Risk": ("PREDICTIVE INTELLIGENCE", "Delivery Risk Intelligence", "Priorize remessas com sinais de atraso e acompanhe a prontidão para previsão."),
    "04 · Modelo & qualidade": ("MODEL GOVERNANCE", "Modelo & qualidade dos dados", "Transparência sobre cobertura, confiabilidade e desempenho analítico."),
}
eyebrow, title, subtitle = page_titles[page]
st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div>', unsafe_allow_html=True)
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ----------------------------- Page 1 -----------------------------
if page == "01 · Visão executiva":
    total = len(filtered)
    delivered = int(filtered["Status_norm"].eq("delivered").sum())
    cancelled = int(filtered["Status_norm"].isin(["cancelled", "canceled"]).sum())
    trustworthy_count = len(trusted)
    ontime_rate = delivered_trusted["entregue_no_prazo"].mean() if len(delivered_trusted) else np.nan
    late_rate = delivered_trusted["entregue_atrasado"].mean() if len(delivered_trusted) else np.nan
    avg_delay = delivered_trusted.loc[delivered_trusted["entregue_atrasado"], "Dias de atraso"].mean() if delivered_trusted["entregue_atrasado"].any() else np.nan
    k = st.columns(6)
    vals = [("ENTREGAS NA BASE", f"{total:,}".replace(",","."), "Registros após filtros", CYAN),
            ("ENTREGUES", f"{delivered:,}".replace(",","."), pct(delivered/total if total else np.nan)+" do volume", GREEN),
            ("CANCELADOS", f"{cancelled:,}".replace(",","."), pct(cancelled/total if total else np.nan)+" do volume", ORANGE),
            ("PROMESSAS CONFIÁVEIS", f"{trustworthy_count:,}".replace(",","."), pct(trustworthy_count/total if total else np.nan)+" da base", BLUE),
            ("PONTUALIDADE VALIDÁVEL", pct(ontime_rate), f"{len(delivered_trusted)} entregas com promessa válida", GREEN),
            ("ATRASO MÉDIO", f"{avg_delay:.1f} dias" if pd.notna(avg_delay) else "—", "Somente entregas atrasadas validadas", RED)]
    for col, args in zip(k, vals):
        with col: card(*args)

    st.markdown('<div class="section-head">Pulso da operação</div>', unsafe_allow_html=True)
    left, right = st.columns([1.55, 1])
    with left:
        monthly = filtered.dropna(subset=["mes_compra"]).groupby("mes_compra", as_index=False).agg(Pedidos=("pedido_ref","count")) if "pedido_ref" in filtered else filtered.dropna(subset=["mes_compra"]).groupby("mes_compra",as_index=False).size().rename(columns={"size":"Pedidos"})
        if not monthly.empty:
            fig = px.area(monthly, x="mes_compra", y="Pedidos", markers=True, title="Volume de pedidos por mês", color_discrete_sequence=[CYAN])
            fig.update_traces(line_width=3, fillcolor="rgba(66,214,199,.13)")
            st.plotly_chart(apply_layout(fig, 330), use_container_width=True)
        else: st.info("Sem datas de compra válidas para a seleção.")
    with right:
        status = filtered["Resultado_norm"].value_counts().rename_axis("Resultado").reset_index(name="Pedidos")
        fig = px.pie(status, names="Resultado", values="Pedidos", hole=.62, title="Distribuição dos resultados", color_discrete_sequence=[CYAN, BLUE, ORANGE, RED, GREEN, "#9276D5"])
        fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color=NAVY,width=2)))
        st.plotly_chart(apply_layout(fig, 330), use_container_width=True)

    left, right = st.columns(2)
    with left:
        uf = filtered.groupby("UF", as_index=False).agg(Pedidos=("UF","size")).sort_values("Pedidos", ascending=False).head(10)
        fig = px.bar(uf.sort_values("Pedidos"), x="Pedidos", y="UF", orientation="h", title="Top 10 UFs por volume", color="Pedidos", color_continuous_scale=["#24415D", CYAN])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_layout(fig, 330), use_container_width=True)
    with right:
        if len(delivered_trusted):
            perf = pd.DataFrame({"Indicador":["No prazo","Atrasadas"],"Pedidos":[int(delivered_trusted["entregue_no_prazo"].sum()),int(delivered_trusted["entregue_atrasado"].sum())]})
            fig = px.bar(perf, x="Indicador", y="Pedidos", color="Indicador", title="Pontualidade — somente promessa confiável", color_discrete_map={"No prazo":GREEN,"Atrasadas":RED})
            fig.update_layout(showlegend=False)
            st.plotly_chart(apply_layout(fig, 330), use_container_width=True)
        else:
            st.info("Ainda não há entregas concluídas com promessa confiável no recorte selecionado.")
    st.markdown('<div class="insight">A taxa de pontualidade considera apenas entregas concluídas com promessa marcada como <b>Confiável</b> e origem <b>captura_inicial</b>. Isso evita avaliar a operação usando prazos reconstruídos ou não validados.</div>', unsafe_allow_html=True)

# ----------------------------- Page 2 -----------------------------
elif page == "02 · Diagnóstico operacional":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Modalidades", f"{filtered['Modalidade_norm'].nunique()}")
    c2.metric("UFs atendidas", f"{filtered['UF'].nunique()}")
    c3.metric("Itens por pedido", f"{filtered['quantidade_itens'].mean():.1f}" if filtered["quantidade_itens"].notna().any() else "—")
    c4.metric("Valor médio dos produtos", money(filtered["valor_produtos"].mean()) if filtered["valor_produtos"].notna().any() else "—")
    st.markdown('<div class="section-head">Onde a operação concentra volume?</div>', unsafe_allow_html=True)
    left,right = st.columns(2)
    with left:
        bymod = filtered.groupby("Modalidade_norm",as_index=False).agg(Pedidos=("Modalidade_norm","size"), Valor_medio=("valor_produtos","mean"))
        fig = px.bar(bymod.sort_values("Pedidos",ascending=False), x="Modalidade_norm", y="Pedidos", color="Modalidade_norm", title="Volume por modalidade", color_discrete_sequence=[CYAN,BLUE,ORANGE,GREEN])
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_layout(fig),use_container_width=True)
    with right:
        byuf = filtered.groupby("UF",as_index=False).agg(Pedidos=("UF","size"),Atraso_medio=("Dias de atraso","mean")).sort_values("Pedidos",ascending=False).head(12)
        fig = px.bar(byuf.sort_values("Pedidos"),x="Pedidos",y="UF",orientation="h",color="Atraso_medio",title="Volume por UF e atraso médio observado",color_continuous_scale=["#24415D",ORANGE,RED],hover_data={"Atraso_medio":":.2f"})
        st.plotly_chart(apply_layout(fig),use_container_width=True)
    st.markdown('<div class="section-head">Fluxo e etapas logísticas</div>', unsafe_allow_html=True)
    stage_cols = [("Compra → despacho","data_compra_dt","Despacho real_dt"),("Despacho → 1ª tentativa","Despacho real_dt","Primeira tentativa_dt"),("1ª tentativa → entrega","Primeira tentativa_dt","Entrega real_dt")]
    stage_rows=[]
    for label,a,b in stage_cols:
        if a in filtered and b in filtered:
            delta=(filtered[b]-filtered[a]).dt.total_seconds()/3600
            delta=delta[(delta>=0)&(delta<=24*60)]
            stage_rows.append({"Etapa":label,"Tempo médio (h)":delta.mean(),"Registros":int(delta.notna().sum())})
    if stage_rows:
        stage=pd.DataFrame(stage_rows)
        fig=px.bar(stage,x="Etapa",y="Tempo médio (h)",text=stage["Tempo médio (h)"].map(lambda x:f"{x:.1f} h" if pd.notna(x) else "—"),title="Tempo médio entre eventos registrados",color="Tempo médio (h)",color_continuous_scale=[CYAN,BLUE,ORANGE])
        fig.update_traces(textposition="outside")
        st.plotly_chart(apply_layout(fig,300),use_container_width=True)
    else: st.info("Não há datas suficientes para calcular tempos entre etapas.")
    st.markdown('<div class="warning">Os tempos entre etapas usam apenas registros com os dois eventos preenchidos. Ausência de evento não é tratada como tempo zero. Médias são descritivas e não provam causalidade.</div>',unsafe_allow_html=True)

# ----------------------------- Page 3 -----------------------------
elif page == "03 · Delivery Risk":
    st.markdown('<div class="warning"><b>Modo atual: monitoramento operacional.</b> O modelo de Machine Learning ainda não está treinado com a base Olist. Portanto, esta página não apresenta scores preditivos fictícios. Os sinais abaixo vêm dos campos observados na planilha.</div>',unsafe_allow_html=True)
    eligible_n=int(open_eligible.shape[0])
    late_n=int(trusted["Resultado_norm"].isin(["Em transito atrasado","Entregue atrasado","Nao entregue"]).sum())
    ontime_open=int(trusted["Resultado_norm"].eq("Em transito no prazo").sum())
    k=st.columns(4)
    with k[0]: card("ELEGÍVEIS PARA FUTURA INFERÊNCIA",f"{eligible_n}","Promessa confiável + campos válidos",BLUE)
    with k[1]: card("ATRASO / NÃO ENTREGA OBSERVADOS",f"{late_n}","Sinais operacionais na base confiável",RED)
    with k[2]: card("EM TRÂNSITO NO PRAZO",f"{ontime_open}","Status observado",GREEN)
    with k[3]: card("MODELO PREDITIVO","ATIVO" if model else "PENDENTE","Treinamento Olist" if model else "Aguardando base Olist",CYAN if model else ORANGE)
    st.markdown('<div class="section-head">Sinais operacionais por resultado</div>',unsafe_allow_html=True)
    risk_base=trusted.copy()
    if not risk_base.empty:
        order=["Em transito atrasado","Nao entregue","Entregue atrasado","Em transito no prazo","Entregue no prazo"]
        risk_counts=risk_base["Resultado_norm"].value_counts().rename_axis("Sinal").reset_index(name="Pedidos")
        risk_counts["ord"] = risk_counts["Sinal"].map({x:i for i,x in enumerate(order)})
        risk_counts=risk_counts.sort_values("ord")
        fig=px.bar(risk_counts,x="Pedidos",y="Sinal",orientation="h",title="Distribuição dos sinais — promessas confiáveis",color="Sinal",color_discrete_map={"Em transito atrasado":RED,"Nao entregue":RED,"Entregue atrasado":ORANGE,"Em transito no prazo":BLUE,"Entregue no prazo":GREEN})
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_layout(fig,300),use_container_width=True)
    else:
        st.info("Nenhum registro com promessa confiável no recorte atual.")
    st.markdown('<div class="section-head">Fila de acompanhamento</div>',unsafe_allow_html=True)
    queue=trusted[trusted["Resultado_norm"].isin(["Em transito atrasado","Nao entregue"])].copy()
    if queue.empty:
        st.success("Nenhum caso com sinal de atraso em aberto no recorte selecionado.")
    else:
        cols=[c for c in ["pedido_ref","UF","Modalidade_norm","data_compra","data_prometida_original","Resultado_norm","Dias de atraso","valor_produtos","valor_frete","Motivo"] if c in queue]
        queue=queue.sort_values("Dias de atraso",ascending=False)
        st.dataframe(queue[cols],use_container_width=True,hide_index=True)
        st.download_button("Exportar fila de acompanhamento (CSV)",queue[cols].to_csv(index=False).encode("utf-8-sig"),"fila_delivery_risk.csv","text/csv")
    if model and metadata:
        st.success(f"Modelo treinado disponível: {metadata.get('model_name','Algoritmo')} · {metadata.get('trained_at_utc','')}")
        st.caption("O painel poderá exibir scores após a inferência da base atual, desde que os registros sejam elegíveis.")
    else:
        st.info("Próxima etapa de ML: adicionar os CSVs oficiais da Olist em data/olist/ e executar `python train.py`. A transferência para operações atuais ainda exigirá validação específica.")

# ----------------------------- Page 4 -----------------------------
else:
    total=len(filtered)
    trusted_n=int(filtered["promessa_confiavel"].sum())
    missing_zip=int(filtered.get("cep_destino",pd.Series(index=filtered.index,dtype="object")).isna().sum())
    missing_purchase=int(filtered.get("data_compra_dt",pd.Series(index=filtered.index,dtype="datetime64[ns]")).isna().sum())
    invalid_promise=int((~filtered["promessa_confiavel"]).sum())
    eligible=int(filtered["eligible_for_prediction"].sum())
    k=st.columns(5)
    for col,(label,value,note) in zip(k,[
        ("REGISTROS",f"{total:,}".replace(",","."),"Após filtros"),
        ("PROMESSA CONFIÁVEL",pct(trusted_n/total if total else np.nan),f"{trusted_n} registros"),
        ("PROMESSA NÃO VALIDADA",pct(invalid_promise/total if total else np.nan),f"{invalid_promise} registros"),
        ("ELEGÍVEIS PARA ML",f"{eligible}","Inferência preventiva"),
        ("CEP AUSENTE",f"{missing_zip}","Campo bruto ausente")]):
        with col: card(label,value,note,ORANGE if label in ["PROMESSA NÃO VALIDADA","CEP AUSENTE"] else CYAN)
    st.markdown('<div class="section-head">Cobertura e qualidade da informação</div>',unsafe_allow_html=True)
    left,right=st.columns(2)
    with left:
        q=pd.DataFrame({"Categoria":["Promessa confiável","Promessa não validada"],"Registros":[trusted_n,invalid_promise]})
        fig=px.pie(q,names="Categoria",values="Registros",hole=.62,title="Confiabilidade da promessa",color="Categoria",color_discrete_map={"Promessa confiável":GREEN,"Promessa não validada":ORANGE})
        fig.update_traces(textinfo="percent+value")
        st.plotly_chart(apply_layout(fig,320),use_container_width=True)
    with right:
        q2=pd.DataFrame({"Campo":["Data da compra ausente","CEP ausente","Promessa não validada","Elegível para inferência"],"Registros":[missing_purchase,missing_zip,invalid_promise,eligible]})
        fig=px.bar(q2,x="Registros",y="Campo",orientation="h",title="Indicadores de cobertura",color="Registros",color_continuous_scale=["#24415D",CYAN])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_layout(fig,320),use_container_width=True)
    st.markdown('<div class="section-head">Governança do modelo</div>',unsafe_allow_html=True)
    if metadata:
        st.success(f"Modelo carregado: {metadata.get('model_name','—')} · Treinado em {metadata.get('trained_at_utc','—')}")
        metrics=metadata.get("test_metrics",{})
        cols=st.columns(5)
        for col,key,label in zip(cols,["pr_auc","precision","recall","f1","top_20pct_lift"],["PR-AUC","Precisão","Recall","F1-score","Lift top 20%"]):
            val=metrics.get(key)
            with col: st.metric(label,"—" if val is None else f"{val:.3f}")
    else:
        st.warning("Modelo ainda não treinado. As métricas de desempenho não serão preenchidas com valores simulados.")
        st.code("# Coloque os CSVs oficiais da Olist em data/olist/\npython train.py",language="bash")
    with st.expander("Regras metodológicas e limitações"):
        st.markdown("""
        - Pontualidade é calculada somente em entregas concluídas com promessa confiável e origem `captura_inicial`.
        - Entrega real, status final e atraso não são usados como atributos preditivos.
        - Pedidos entregues/cancelados não entram na fila de previsão preventiva.
        - Olist (2016–2018) não representa diretamente as modalidades Flex, Agência e Full atuais.
        - Sem validação temporal e teste independente, não há alegação de desempenho preditivo.
        - O score do modelo, quando disponível, é tratado como ranking de risco; não como probabilidade calibrada sem avaliação de calibração.
        """)

st.markdown("<hr>",unsafe_allow_html=True)
st.markdown(f"<div style='display:flex;justify-content:space-between;gap:12px;color:{MUTED};font-size:.72rem'><span>BREAKING BAD · DELIVERY RISK INTELLIGENCE</span><span>POC ANALÍTICA · DADOS PROCESSADOS LOCALMENTE</span></div>",unsafe_allow_html=True)
