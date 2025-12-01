
# app.py
# --------------------------------------------
# Ultra Preditor - Flask (produção / Render)
# --------------------------------------------
# Observações:
# - Use gunicorn para iniciar no Render: "gunicorn app:app"
# - Coloque index.html dentro de /templates
# - Mantenha os modelos .pkl na mesma pasta do app.py
# - Adicione as libs no requirements.txt (inclua openpyxl)

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')  # backend sem GUI (necessário em servidores)
import matplotlib.pyplot as plt
import io
import base64

# ------------------------------------------------------------------------------
# Configuração básica do app e logs
# ------------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------
# Carregamento dos modelos
# ------------------------------------------------------------------------------
def _carregar_modelo(caminho_arquivo):
    """Carrega um modelo .pkl com tratamento de erros e logs."""
    caminho = os.path.join(BASE_DIR, caminho_arquivo)
    if not os.path.exists(caminho):
        logger.error(f"Arquivo de modelo não encontrado: {caminho_arquivo}")
        raise FileNotFoundError(f"Modelo '{caminho_arquivo}' não encontrado no servidor.")
    logger.info(f"Carregando modelo: {caminho_arquivo}")
    return joblib.load(caminho)

# Modelos e features (ajuste os nomes dos arquivos se necessário)
modelo_info = _carregar_modelo('modelo_randomforest.pkl')
modelo = modelo_info['modelo']
features = modelo_info['features']

modelo_agregadores_info = _carregar_modelo('modelo_agregadores.pkl')
modelo_agregadores = modelo_agregadores_info['modelo']
features_agregadores = modelo_agregadores_info['features']

# ------------------------------------------------------------------------------
# Configurações e mapeamentos
# ------------------------------------------------------------------------------
# ATENÇÃO: "Vagas " tem um espaço no final porque provavelmente é assim no treinamento.
colunas_manuais = ['Vagas ', 'Metragem', 'Quantidade de Concorrentes']

# Mapeia o nome da feature usada no modelo para palavras-chave que podem aparecer na planilha
mapa_colunas = {
    'Renda média domiciliar': ['renda média', 'renda media'],
    'População Mulheres': ['mulheres'],
    'População Homens': ['homens'],
    'PEA Dia': ['pea'],
    ' de 20 a 24 anos': ['20 a 24'],
    'População': ['população'],
    'Densidade demográfica': ['densidade']
}

# Métricas de referência (ex.: val. de treino/validação do seu modelo)
METRICAS_MODELO = {'r2': 0.8944, 'rmse': 200.44, 'mae': 140.27}
METRICAS_AGREGADORES = {'r2': 0.8410, 'rmse': 303.92, 'mae': 185.20}

# ------------------------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------------------------
def criar_grafico_comparativo(
    previsao_recorrentes: float,
    previsao_agregadores: float,
    metricas_recorrentes: dict,
    metricas_agregadores: dict
) -> str:
    """
    Cria um gráfico de barras comparando mínimos, previsão e máximos
    para recorrentes e agregadores. Retorna a imagem em base64 para embutir no HTML.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Dados
    categorias = ['Mínimo', 'Previsão', 'Máximo']
    valores_rec = [
        metricas_recorrentes['min_previsto'],
        previsao_recorrentes,
        metricas_recorrentes['max_previsto']
    ]
    valores_agr = [
        metricas_agregadores['min_previsto'],
        previsao_agregadores,
        metricas_agregadores['max_previsto']
    ]

    cores_rec = ['#FF6B6B', '#EE5A24', '#00B894']
    cores_agr = ['#4ECDC4', '#45B7D1', '#96CEB4']

    # Gráfico Recorrentes
    barras1 = ax1.bar(categorias, valores_rec, color=cores_rec, alpha=0.85)
    ax1.set_title('Alunos Recorrentes', fontsize=14, fontweight='bold', color='#2c3e50')
    ax1.set_ylabel('Quantidade de Alunos', fontsize=11, fontweight='bold')

    # Valores nas barras
    for i, (b, v) in enumerate(zip(barras1, valores_rec)):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(valores_rec) * 0.02,
            f'{v:.0f}',
            ha='center', va='bottom',
            fontweight='bold', fontsize=12,
            color=cores_rec[i]
        )

    # Gráfico Agregadores
    barras2 = ax2.bar(categorias, valores_agr, color=cores_agr, alpha=0.85)
    ax2.set_title('Alunos Agregadores', fontsize=14, fontweight='bold', color='#2c3e50')
    ax2.set_ylabel('Quantidade de Alunos', fontsize=11, fontweight='bold')

    for i, (b, v) in enumerate(zip(barras2, valores_agr)):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(valores_agr) * 0.02,
            f'{v:.0f}',
            ha='center', va='bottom',
            fontweight='bold', fontsize=12,
            color=cores_agr[i]
        )

    # Visual final
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(True, alpha=0.2, axis='y', color='#bdc3c7')
        ax.grid(False, axis='x')
        ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    # Salva em memória e retorna base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode()

def calcular_metricas(previsao: float, tipo_modelo: str = 'recorrentes') -> dict:
    """
    Calcula intervalo de confiança aproximado com base no RMSE.
    Retorna dicionário com métricas e intervalo.
    """
    base = METRICAS_MODELO if tipo_modelo == 'recorrentes' else METRICAS_AGREGADORES
    margem_erro = base['rmse'] * 1.5  # ajuste fino conforme sua calibração

    min_prev = max(0.0, float(previsao) - margem_erro)
    max_prev = float(previsao) + margem_erro

    metricas = {
        'previsao': float(previsao),
        'min_previsto': min_prev,
        'max_previsto': max_prev,
        'intervalo_confianca': f"{min_prev:.0f} - {max_prev:.0f}",
        'margem_erro': margem_erro,
        'r2': base['r2'],
        'rmse': base['rmse'],
        'mae': base['mae'],
        'acuracia_percentual': f"{(base['r2'] * 100):.1f}%"
    }
    return metricas

def _buscar_valor_planilha(df: pd.DataFrame, feature_nome: str) -> float:
    """
    Encontra o valor correspondente a uma feature em df usando mapa_colunas.
    Se não achar por palavras-chave, tenta por igualdade aproximada.
    Retorna 0.0 se não encontrar.
    """
    if df is None or df.empty:
        return 0.0

    palavras = mapa_colunas.get(feature_nome, [])
    coluna_encontrada = None

    # Busca por palavras-chave (contains, case-insensitive)
    for col in df.columns:
        if any(p.lower() in str(col).lower() for p in palavras):
            coluna_encontrada = col
            break

    # Fallback: busca por igualdade aproximada do nome
    if coluna_encontrada is None:
        for col in df.columns:
            if feature_nome.strip().lower() == str(col).strip().lower():
                coluna_encontrada = col
                break

    try:
        valor = df[coluna_encontrada].iloc[0] if coluna_encontrada else 0
        return float(valor) if pd.notnull(valor) else 0.0
    except Exception as e:
        logger.warning(f"Falha ao ler coluna '{coluna_encontrada}': {e}")
        return 0.0

def preparar_dados_entrada(
    df: pd.DataFrame,
    features_necessarias: list,
    vagas: float,
    metragem: float,
    concorrentes: float
) -> list:
    """
    Prepara o vetor de entrada respeitando a ordem de 'features_necessarias'.
    Para colunas manuais, usa os valores do formulário.
    Para demais, tenta extrair da planilha usando _buscar_valor_planilha().
    """
    valores = []
    for feature in features_necessarias:
        if feature in colunas_manuais:
            if feature == 'Vagas ':
                valores.append(float(vagas))
            elif feature == 'Metragem':
                valores.append(float(metragem))
            elif feature == 'Quantidade de Concorrentes':
                valores.append(float(concorrentes))
        else:
            valores.append(_buscar_valor_planilha(df, feature))
    return valores

# ------------------------------------------------------------------------------
# Rotas
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    # Certifique-se de que templates/index.html existe
    return render_template("index.html")

@app.route("/health")
def health():
    # Útil para o Render verificar se a aplicação está de pé
    return jsonify({"status": "ok"}), 200

@app.route("/prever", methods=["POST"])
def prever():
    """
    Rota que recebe:
      - campos do formulário: 'vagas', 'metragem', 'concorrentes'
      - arquivo Excel opcional: 'planilha' (.xlsx)
    Retorna JSON com previsões, métricas e gráfico base64.
    """
    try:
        # Validação básica dos inputs
        vagas = float(request.form.get("vagas", "0") or 0)
        metragem = float(request.form.get("metragem", "0") or 0)
        concorrentes = float(request.form.get("concorrentes", "0") or 0)

        # Processa planilha se enviada
        planilha = request.files.get("planilha")
        df = None
        if planilha and planilha.filename.lower().endswith(".xlsx"):
            # engine "openpyxl" precisa estar no requirements
            df = pd.read_excel(planilha, engine="openpyxl")
            logger.info(f"Planilha recebida com {len(df.columns)} colunas.")

        # Monta entrada e prevê - Recorrentes
        valores_rec = preparar_dados_entrada(df, features, vagas, metragem, concorrentes)
        dados_rec = pd.DataFrame([valores_rec], columns=features)
        previsao_rec = float(modelo.predict(dados_rec)[0])

        # Monta entrada e prevê - Agregadores
        valores_agr = preparar_dados_entrada(df, features_agregadores, vagas, metragem, concorrentes)
        dados_agr = pd.DataFrame([valores_agr], columns=features_agregadores)
        previsao_agr = float(modelo_agregadores.predict(dados_agr)[0])

        # Métricas
        metricas_rec = calcular_metricas(previsao_rec, "recorrentes")
        metricas_agr = calcular_metricas(previsao_agr, "agregadores")

        # Gráfico
        grafico_b64 = criar_grafico_comparativo(
            previsao_rec, previsao_agr, metricas_rec, metricas_agr
        )

        total = previsao_rec + previsao_agr

        resposta = {
            "status": "success",
            "mensagem": "Previsões calculadas com sucesso!",
            "previsao_recorrentes": f"{previsao_rec:.0f}",
            "previsao_agregadores": f"{previsao_agr:.0f}",
            "total_geral": f"{total:.0f}",
            "metricas_recorrentes": metricas_rec,
            "metricas_agregadores": metricas_agr,
            "grafico": grafico_b64
        }
        return jsonify(resposta), 200

    except Exception as e:
        logger.exception("Erro na rota /prever")
        return jsonify({"status": "error", "error": str(e)}), 500

# ------------------------------------------------------------------------------
# IMPORTANTE:
# - NÃO execute app.run() em produção (Render usará gunicorn).
# ------------------------------------------------------------------------------
# if __name__ == "__main__":
