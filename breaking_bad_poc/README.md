# Breaking Bad — Delivery Risk Intelligence (MVP/POC)

Aplicação local para validar arquivos de pedidos, preparar dados e priorizar pedidos por risco estimado de atraso. O projeto segue o relatório CRISP-DM do grupo: Olist para treinamento/teste; exportação da empresa para inferência futura, sem integração com ERP/API.

> **Estado desta entrega:** a aplicação e o pipeline estão implementados. O pacote de nove CSVs da Olist não foi incluído nos arquivos recebidos, portanto nenhum modelo real foi treinado ou desempenho preditivo alegado. Sem artefato treinado, o app valida os pedidos e informa claramente que não pode pontuá-los.

## 1. Requisitos
- Python 3.11+
- Windows, Linux ou macOS
- Dependências em `requirements.txt`

## 2. Executar localmente (Windows PowerShell)
```powershell
cd breaking_bad_poc
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Acesse `http://localhost:8501`. Para habilitar senha no app:
```powershell
$env:APP_PASSWORD = "uma-senha-forte"
streamlit run app.py
```
Sem `APP_PASSWORD`, o app avisa que está em modo local sem autenticação. Não exponha a porta à rede.

## 3. Treinar com a Olist
Baixe a distribuição oficial Olist (Kaggle) e extraia os nove CSVs em `data/olist/`, mantendo os nomes originais. O pipeline exige ao menos:
- `olist_orders_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_order_items_dataset.csv`

Os outros CSVs fazem parte da distribuição/documentação, mas não são necessários para as features iniciais. Preserve os originais e observe a licença CC BY-NC-SA 4.0.

```powershell
python train.py
```
O treinamento agrega itens por `order_id`, seleciona pedidos entregues com datas válidas, calcula atraso por dia de calendário, ordena cronologicamente e separa 70% treino, 15% validação e 15% teste. Compara Regressão Logística e Random Forest pela PR-AUC de validação; refaz o escolhido em treino+validação e avalia uma única vez no teste. Artefatos são gravados em `models/`.

**Segurança:** `joblib`/pickle pode executar código ao carregar artefatos maliciosos. O app só carrega o artefato criado localmente pelo script do projeto. Nunca coloque modelo de terceiros no diretório `models/`.

## 4. Importar a planilha da empresa
Na interface, escolha `Importar pedidos` e envie CSV ou XLSX com a estrutura da aba `Entregas`. O contrato exige as colunas:
`pedido_ref`, `data_compra`, `data_prometida_original`, `Promessa confiavel`, `Origem da promessa`, `Status do envio`, `uf_destino`, `quantidade_itens`, `valor_produtos`, `valor_frete`, `cep_destino`.

Somente pedidos abertos com promessa `Confiavel` e origem `captura_inicial`, datas coerentes, UF, CEP, quantidade e valores válidos serão elegíveis. Pedidos entregues/cancelados não são previsões preventivas. Campos de entrega real, resultado e atraso não entram no modelo.

O CEP é convertido em texto e reduzido a prefixo de cinco dígitos. Referências são exibidas de forma limitada e não usadas como feature. Reimportações removem referências duplicadas no lote (primeira ocorrência) e reportam a quantidade.

## 5. Docker
```bash
cp .env.example .env
# Edite APP_PASSWORD com senha forte
docker compose up --build
```
A publicação está vinculada a `127.0.0.1` no host. O container roda como usuário não-root. A pasta de Olist é montada somente para leitura.

## 6. Testes
```bash
pytest -q
```

## 7. Estrutura
```text
app.py                         Interface Streamlit
train.py                       Entrada do treinamento
src/breaking_bad/config.py      Contrato e configuração
src/breaking_bad/security.py    Validação de upload
src/breaking_bad/ingestion.py   Leitura, normalização e elegibilidade
src/breaking_bad/training.py    ETL Olist, pipeline, avaliação e persistência
tests/                         Testes automatizados
data/olist/                    CSVs oficiais (não versionar)
models/                        Artefatos locais (não versionar)
```

## 8. Próximas melhorias antes de uso operacional
- Confirmar equivalência semântica e unidade dos campos locais e Olist.
- Verificar cobertura postal aprendida no conjunto de treino, além de UF fallback se treinada e avaliada.
- Calibração de probabilidades e escolha do limiar na validação.
- Registro imutável de previsões, data/hora, versão do modelo e desfechos para avaliação local consentida.
- Autenticação multiusuário e trilha de auditoria se houver implantação compartilhada.
- Política formal de retenção, backup e descarte dos dados privados.
- Avaliação prospectiva em pedidos novos com promessa original comprovada.

## 9. Limites
Olist (2016–2018) não contém modalidades Flex, Agência ou Full. A pontuação não comprova desempenho no Mercado Livre atual. A aplicação não envia mensagens, não altera prazos, não despacha pedidos e não se conecta ao ERP ou API. O resultado deve ser tratado como apoio à decisão humana, nunca como certeza de atraso.
