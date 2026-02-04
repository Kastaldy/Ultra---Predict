
# app.py — Ultra Preditor (produção / Render)
# Observações:
# - requirements.txt: Flask, pandas, joblib, numpy, matplotlib, gunicorn, openpyxl, scikit-learn, unidecode

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import unicodedata
from typing import Dict, List, Tuple

app = Flask(__name__, template_folder="templates")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------
# Carregamento dos modelos 
# ------------------------------------------------------------------------------
def _carregar_modelo(caminho_arquivo: str):
    caminho = os.path.join(BASE_DIR, caminho_arquivo)
    if not os.path.exists(caminho):
        logger.error(f"Arquivo de modelo não encontrado: {caminho_arquivo}")
        raise FileNotFoundError(f"Modelo '{caminho_arquivo}' não encontrado no servidor.")
    logger.info(f"Carregando modelo: {caminho_arquivo}")
    return joblib.load(caminho)

modelo_info = _carregar_modelo('modelo_randomforest.pkl')
modelo = modelo_info['modelo']
features = modelo_info['features']

modelo_agregadores_info = _carregar_modelo('modelo_agregadores.pkl')
modelo_agregadores = modelo_agregadores_info['modelo']
features_agregadores = modelo_agregadores_info['features']

# ------------------------------------------------------------------------------
# Configurações e mapeamentos
# ------------------------------------------------------------------------------

colunas_manuais = ['Vagas ', 'Metragem', 'Quantidade de Concorrentes']

# Palavras-chave para encontrar cada feature demográfica na planilha
mapa_colunas = {
    'Renda média domiciliar': ['renda média domiciliar', 'renda media domiciliar', 'renda média'],
    'População Mulheres':    ['população mulheres', 'mulheres'],
    'População Homens':      ['população homens', 'homens'],
    'PEA Dia':               ['pea dia', 'pea'],
    ' de 20 a 24 anos':      ['20 a 24', 'de 20 a 24 anos'],
    'População':             ['população'],
    'Densidade demográfica': ['densidade demográfica', 'densidade demografica', 'densidade']
}


METRICAS_MODELO       = {'r2': 0.8944, 'rmse': 200.44, 'mae': 140.27}
METRICAS_AGREGADORES  = {'r2': 0.8410, 'rmse': 263.92, 'mae': 185.20}


OBRIGATORIAS = [
    'Renda média domiciliar',
    'População',
    'População Homens',
    'População Mulheres',
    'PEA Dia',
    'Densidade demográfica',
    ' de 20 a 24 anos'
]

# ------------------------------------------------------------------------------
# Utilidades de normalização e parsing da planilha
# ------------------------------------------------------------------------------
def _normalize(text: str) -> str:
    """remove acentos, deixa minúsculo e compacta espaços."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize('NFKD', text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower().strip()
    t = " ".join(t.split())
    return t

def _planilha_para_dict(df: pd.DataFrame) -> Dict[str, float]:
    """
    Converte uma planilha 'vertical' (rótulo em uma linha, valor na linha seguinte)
    ou uma planilha de duas colunas (nome/valor) em um dicionário {nome_normalizado: valor}.
    """
    pares: Dict[str, float] = {}
    if df is None or df.empty:
        return pares

    # Remove linhas/colunas totalmente vazias
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)

    # Caso 1: existe coluna de rótulo + coluna numérica (mais comum)
    if df.shape[1] >= 2:
       
        valor_idx = None
        for j in range(1, min(4, df.shape[1])):
            if pd.api.types.is_numeric_dtype(df.iloc[:, j]):
                valor_idx = j
                break
       
        if valor_idx is None:
            valor_idx = 1

        rotulos = df.iloc[:, 0]
        valores = df.iloc[:, valor_idx]
        for r, v in zip(rotulos, valores):
            if isinstance(r, str) and pd.notnull(v):
                try:
                    pares[_normalize(r)] = float(v)
                except Exception:
                   
                    s = str(v).replace('.', '').replace(',', '.')
                    try:
                        pares[_normalize(r)] = float(s)
                    except Exception:
                        continue
        return pares

   
    if df.shape[1] == 1:
        col = df.iloc[:, 0].tolist()
        for i in range(len(col) - 1):
            nome, prox = col[i], col[i + 1]
            if isinstance(nome, str) and (pd.api.types.is_number(prox) or _normalize(str(prox))):
                try:
                    valor = float(prox)
                except Exception:
                    s = str(prox).replace('.', '').replace(',', '.')
                    try:
                        valor = float(s)
                    except Exception:
                        continue
                pares[_normalize(nome)] = valor
        return pares

    return pares

def _buscar_valor(pares: Dict[str, float], feature_nome: str) -> float:
    """busca pelo valor no dicionário usando mapa_colunas e fallback para igualdade aproximada."""
    palavras = mapa_colunas.get(feature_nome, [])

    for chave_norm, valor in pares.items():
        for p in palavras:
            if _normalize(p) in chave_norm:
                return float(valor)
  
    alvo = _normalize(feature_nome)
    for chave_norm, valor in pares.items():
        if chave_norm == alvo:
            return float(valor)
    return 0.0

# ------------------------------------------------------------------------------
# Funções de modelo e visualização
# ------------------------------------------------------------------------------
def criar_grafico_comparativo(previsao_recorrentes, previsao_agregadores,
                              metricas_recorrentes, metricas_agregadores) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    categorias = ['Mínimo', 'Previsão', 'Máximo']
    valores_rec = [metricas_recorrentes['min_previsto'], previsao_recorrentes, metricas_recorrentes['max_previsto']]
    valores_agr = [metricas_agregadores['min_previsto'], previsao_agregadores, metricas_agregadores['max_previsto']]

    cores_rec = ['#FF6B6B', '#EE5A24', '#00B894']
    cores_agr = ['#4ECDC4', '#45B7D1', '#96CEB4']

    b1 = ax1.bar(categorias, valores_rec, color=cores_rec, alpha=0.85)
    ax1.set_title('Alunos Recorrentes', fontsize=14, fontweight='bold', color='#2c3e50')
    ax1.set_ylabel('Quantidade de Alunos', fontsize=11, fontweight='bold')
    for i, (barra, val) in enumerate(zip(b1, valores_rec)):
        ax1.text(barra.get_x() + barra.get_width()/2, barra.get_height() + max(valores_rec)*0.02,
                 f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12, color=cores_rec[i])

    b2 = ax2.bar(categorias, valores_agr, color=cores_agr, alpha=0.85)
    ax2.set_title('Alunos Agregadores', fontsize=14, fontweight='bold', color='#2c3e50')
    ax2.set_ylabel('Quantidade de Alunos', fontsize=11, fontweight='bold')
    for i, (barra, val) in enumerate(zip(b2, valores_agr)):
        ax2.text(barra.get_x() + barra.get_width()/2, barra.get_height() + max(valores_agr)*0.02,
                 f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12, color=cores_agr[i])

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(True, alpha=0.2, axis='y', color='#bdc3c7')
        ax.grid(False, axis='x')
        ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def calcular_metricas(previsao: float, tipo_modelo: str = 'recorrentes') -> dict:
    base = METRICAS_MODELO if tipo_modelo == 'recorrentes' else METRICAS_AGREGADORES
    margem_erro = base['rmse'] * 1.5
    min_prev = max(0.0, float(previsao) - margem_erro)
    max_prev = float(previsao) + margem_erro
    return {
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

def preparar_dados_entrada(demografia: Dict[str, float], features_necessarias: List[str],
                           vagas: float, metragem: float, concorrentes: float) -> Tuple[List[float], List[str]]:
    """
    Monta o vetor de entrada e retorna também uma lista de features demográficas ausentes.
    """
    valores = []
    faltantes = []
    for feature in features_necessarias:
        if feature in colunas_manuais:
            if feature == 'Vagas ':
                valores.append(float(vagas))
            elif feature == 'Metragem':
                valores.append(float(metragem))
            elif feature == 'Quantidade de Concorrentes':
                valores.append(float(concorrentes))
        else:
            v = _buscar_valor(demografia, feature)
            valores.append(float(v))
            if feature in OBRIGATORIAS and v == 0.0:
                faltantes.append(feature)
    return valores, faltantes

# ------------------------------------------------------------------------------
# Rotas
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/prever", methods=["POST"])
def prever():
    try:
        vagas = float(request.form.get("vagas", "0") or 0)
        metragem = float(request.form.get("metragem", "0") or 0)
        concorrentes = float(request.form.get("concorrentes", "0") or 0)

        planilha = request.files.get("planilha")
        if not planilha or not planilha.filename.lower().endswith(".xlsx"):
            return jsonify({
                "status": "error",
                "mensagem": "É necessário enviar a planilha demográfica (.xlsx)."
            }), 400

       
        df = pd.read_excel(planilha, engine="openpyxl", header=None)
        demografia = _planilha_para_dict(df)
        logger.info(f"Chaves demográficas carregadas: {len(demografia)}")

       
        valores_rec, faltantes_rec = preparar_dados_entrada(demografia, features, vagas, metragem, concorrentes)
       
        valores_agr, faltantes_agr = preparar_dados_entrada(demografia, features_agregadores, vagas, metragem, concorrentes)

        
        faltas_obrig = sorted(set(faltantes_rec + faltantes_agr))
        if faltas_obrig:
            return jsonify({
                "status": "error",
                "mensagem": "Planilha demográfica não possui todas as variáveis obrigatórias.",
                "faltando": faltas_obrig,
                "exemplo_chaves_encontradas": list(demografia.keys())[:20]
            }), 400

        dados_rec = pd.DataFrame([valores_rec], columns=features)
        previsao_rec = float(modelo.predict(dados_rec)[0])

        dados_agr = pd.DataFrame([valores_agr], columns=features_agregadores)
        previsao_agr = float(modelo_agregadores.predict(dados_agr)[0])

        metricas_rec = calcular_metricas(previsao_rec, "recorrentes")
        metricas_agr = calcular_metricas(previsao_agr, "agregadores")

        grafico_b64 = criar_grafico_comparativo(previsao_rec, previsao_agr, metricas_rec, metricas_agr)
        total = previsao_rec + previsao_agr

        return jsonify({
            "status": "success",
            "mensagem": "Previsões calculadas com sucesso!",
            "previsao_recorrentes": f"{previsao_rec:.0f}",
            "previsao_agregadores": f"{previsao_agr:.0f}",
            "total_geral": f"{total:.0f}",
            "metricas_recorrentes": metricas_rec,
            "metricas_agregadores": metricas_agr,
            "grafico": grafico_b64
        }), 200

    except Exception as e:
        logger.exception("Erro na rota /prever")
        return jsonify({"status": "error", "error": str(e)}), 500





