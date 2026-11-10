from io import BytesIO
from pathlib import Path
import zipfile

from .config import MAX_UPLOAD_BYTES, MAX_ROWS


def validate_upload(uploaded_file) -> bytes:
    """Validate size, extension and file signature before parsing."""
    name = Path(uploaded_file.name).name
    suffix = Path(name).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise ValueError("Formato não permitido. Envie somente CSV ou XLSX.")
    raw = uploaded_file.getvalue()
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Arquivo vazio ou acima do limite de 10 MB.")
    if suffix == ".xlsx":
        if not raw.startswith(b"PK\x03\x04"):
            raise ValueError("O conteúdo não corresponde a um arquivo XLSX válido.")
        try:
            with zipfile.ZipFile(BytesIO(raw)) as zf:
                names = zf.namelist()
                if "xl/workbook.xml" not in names or len(names) > 500:
                    raise ValueError("Estrutura XLSX inválida.")
                expanded = sum(i.file_size for i in zf.infolist())
                if expanded > 50 * 1024 * 1024:
                    raise ValueError("Planilha excede o limite de descompactação.")
        except zipfile.BadZipFile as exc:
            raise ValueError("Arquivo XLSX corrompido.") from exc
    elif b"\x00" in raw[:4096]:
        raise ValueError("CSV contém bytes nulos e não será processado.")
    return raw


def enforce_row_limit(df):
    if len(df) > MAX_ROWS:
        raise ValueError(f"Limite de {MAX_ROWS:,} linhas excedido.")
    return df
