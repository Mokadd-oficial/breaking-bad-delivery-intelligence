from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "delivery_risk_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 50_000

LOCAL_REQUIRED = [
    "pedido_ref", "data_compra", "data_prometida_original", "Promessa confiavel",
    "Origem da promessa", "Status do envio", "uf_destino", "quantidade_itens",
    "valor_produtos", "valor_frete", "cep_destino"
]

FEATURES = [
    "purchase_month", "purchase_weekday", "purchase_hour", "promised_days",
    "customer_state", "zip_prefix", "item_count", "total_price", "total_freight"
]
NUMERIC_FEATURES = [
    "purchase_month", "purchase_weekday", "purchase_hour", "promised_days",
    "item_count", "total_price", "total_freight"
]
CATEGORICAL_FEATURES = ["customer_state", "zip_prefix"]
