from io import BytesIO
import pandas as pd

from .config import LOCAL_REQUIRED
from .security import validate_upload, enforce_row_limit


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    raw = validate_upload(uploaded_file)
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx"):
        # data_only=True reads cached values; no formulas or macros are executed.
        df = pd.read_excel(BytesIO(raw), sheet_name="Entregas", engine="openpyxl")
    else:
        try:
            df = pd.read_csv(BytesIO(raw), dtype={"cep_destino": "string"}, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(raw), dtype={"cep_destino": "string"}, encoding="latin-1")
    df.columns = [str(c).strip() for c in df.columns]
    enforce_row_limit(df)
    missing = [c for c in LOCAL_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Colunas obrigatórias ausentes: " + ", ".join(missing))
    return df


def normalize_cep(value):
    if pd.isna(value):
        return None
    # Excel may coerce CEP to float; preserve leading zeros by zero-padding.
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    # Numeric Excel cells may have lost a leading zero; a 7-digit numeric CEP
    # is restored to 8 digits. Arbitrary short text is rejected.
    if len(digits) == 7 and isinstance(value, (int, float)):
        digits = digits.zfill(8)
    if len(digits) != 8:
        return None
    return digits


def local_feature_frame(df: pd.DataFrame):
    out = df.copy()
    out["purchase_dt"] = pd.to_datetime(out["data_compra"], format="%d/%m/%Y %H:%M", errors="coerce")
    out["promise_dt"] = pd.to_datetime(out["data_prometida_original"], format="%d/%m/%Y %H:%M", errors="coerce")
    out["cep_normalized"] = out["cep_destino"].map(normalize_cep)
    out["zip_prefix"] = out["cep_normalized"].map(lambda x: x[:5] if isinstance(x, str) else None)
    out["customer_state"] = out["uf_destino"].astype("string").str.upper().str.strip()
    out["item_count"] = pd.to_numeric(out["quantidade_itens"], errors="coerce")
    out["total_price"] = pd.to_numeric(out["valor_produtos"], errors="coerce")
    out["total_freight"] = pd.to_numeric(out["valor_frete"], errors="coerce")
    out["promised_days"] = (out["promise_dt"].dt.normalize() - out["purchase_dt"].dt.normalize()).dt.days
    out["purchase_month"] = out["purchase_dt"].dt.month
    out["purchase_weekday"] = out["purchase_dt"].dt.weekday
    out["purchase_hour"] = out["purchase_dt"].dt.hour
    return out


def classify_local_rows(df: pd.DataFrame):
    x = local_feature_frame(df)
    status = x["Status do envio"].astype("string").str.lower().fillna("")
    is_open = ~status.isin({"delivered", "canceled", "cancelled", "unavailable"})
    promise_ok = (
        x["Promessa confiavel"].astype("string").str.lower().eq("confiavel")
        & x["Origem da promessa"].astype("string").str.lower().eq("captura_inicial")
    )
    valid = (
        x["purchase_dt"].notna() & x["promise_dt"].notna()
        & x["promised_days"].ge(0) & x["customer_state"].str.fullmatch(r"[A-Z]{2}", na=False)
        & x["zip_prefix"].notna() & x["item_count"].gt(0)
        & x["total_price"].ge(0) & x["total_freight"].ge(0)
    )
    x["eligibility_reason"] = "Elegível para inferência"
    x.loc[~promise_ok, "eligibility_reason"] = "Promessa não confirmada como captura inicial"
    x.loc[promise_ok & ~valid, "eligibility_reason"] = "Campos obrigatórios ausentes ou inválidos"
    x.loc[promise_ok & valid & ~is_open, "eligibility_reason"] = "Pedido concluído/cancelado: não é previsão preventiva"
    x["eligible_for_prediction"] = promise_ok & valid & is_open
    # IDs are retained only for display/deduplication, never included in model features.
    x["display_id"] = x["pedido_ref"].astype("string").fillna("pedido-sem-referencia").str.slice(0, 40)
    return x
