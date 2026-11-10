# Breaking Bad — Delivery Risk Intelligence

Aplicação local de analytics logístico com quatro páginas interativas, inspirada em um dashboard executivo de Power BI. Desenvolvida em Streamlit + Plotly, com processamento em Python e estrutura preparada para integração futura com o modelo de Machine Learning.

## O que está incluído

1. **Visão executiva** — KPIs, volume mensal, distribuição de resultados, ranking de UFs e pontualidade validável.
2. **Diagnóstico operacional** — modalidade, distribuição regional, volume por UF e tempos entre etapas logísticas.
3. **Delivery Risk Intelligence** — sinais operacionais de atraso, fila de acompanhamento e prontidão para inferência.
4. **Modelo & qualidade** — confiabilidade das promessas, qualidade dos dados, elegibilidade e métricas do modelo quando treinado.


## Como executar no Windows

### Opção mais simples

1. Extraia o ZIP em uma pasta.
2. Entre na pasta `breaking_bad_poc`.
3. Dê duplo clique em `INICIAR_DASHBOARD.bat`.
4. Aguarde a instalação das dependências e a abertura do navegador em `http://localhost:8501`.

### Opção pelo terminal

Requer Python 3.11 ou superior.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Trocar a base de demonstração

Na barra lateral, use **Atualizar base (XLSX ou CSV)**. A planilha precisa conter a aba `Entregas` quando for XLSX e as colunas do contrato descrito em `src/breaking_bad/config.py`. O arquivo carregado é processado em memória na sessão e não é enviado a nenhum serviço externo pela aplicação.

## Treinar o modelo

O pacote não contém os dados oficiais da Olist. Para treinar:

1. Coloque os CSVs oficiais necessários em `data/olist/`:
   - `olist_orders_dataset.csv`
   - `olist_customers_dataset.csv`
   - `olist_order_items_dataset.csv`
2. Execute:

```powershell
python train.py
```

O pipeline compara Regressão Logística e Random Forest usando divisão temporal, seleciona pela PR-AUC na validação e avalia no conjunto de teste separado. Os artefatos são salvos em `models/`.

**Limitação metodológica:** Olist (2016–2018) não contém modalidades Flex, Agência ou Full. Mesmo com o modelo treinado, a transferência para uma operação atual exige validação específica. O score deve ser interpretado como ranking de risco, não como probabilidade calibrada sem avaliação de calibração.

## Segurança e privacidade

- Aplicação executada localmente; sem integração com ERP ou API do marketplace.
- Upload limitado a CSV/XLSX, com validação de formato, tamanho e quantidade de linhas.
- Arquivos carregados permanecem na sessão; não são persistidos pelo dashboard.
- Não exponha a porta do Streamlit à rede sem autenticação e revisão de implantação.
- Não use **Publicar na Web** do Power BI para dados operacionais privados.
- `joblib`/pickle pode executar código ao carregar artefatos maliciosos. Use somente modelos treinados localmente pelo pipeline deste projeto.

## Testes

```powershell
python -m pytest -q
```

Os testes existentes cobrem regras básicas de normalização de CEP e elegibilidade. A interface deve ser validada no Windows após a instalação das dependências.
