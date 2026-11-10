import pandas as pd
from src.breaking_bad.ingestion import normalize_cep, classify_local_rows


def test_cep_preserves_leading_zero():
    assert normalize_cep("05508000") == "05508000"
    assert normalize_cep(5508000.0) == "05508000"
    assert normalize_cep("123") is None


def test_future_delivery_fields_are_not_required_for_scoring():
    df = pd.DataFrame([{
        "pedido_ref": "p-1", "data_compra": "10/11/2026 10:00",
        "data_prometida_original": "12/11/2026 00:00", "Promessa confiavel": "Confiavel",
        "Origem da promessa": "captura_inicial", "Status do envio": "shipped",
        "uf_destino": "SP", "quantidade_itens": 1, "valor_produtos": 20,
        "valor_frete": 0, "cep_destino": "05508000"
    }])
    result = classify_local_rows(df)
    assert bool(result.iloc[0].eligible_for_prediction)
    assert result.iloc[0].zip_prefix == "05508"
