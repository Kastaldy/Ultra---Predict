from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Importante para Flask
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Carregar modelos
modelo_info = joblib.load('modelo_randomforest.pkl')
modelo = modelo_info['modelo']
features = modelo_info['features']

# Carregar modelo de agregadores
modelo_agregadores_info = joblib.load('modelo_agregadores.pkl')
modelo_agregadores = modelo_agregadores_info['modelo']
features_agregadores = modelo_agregadores_info['features']

# Configurações
colunas_manuais = ['Vagas ', 'Metragem', 'Quantidade de Concorrentes']
mapa_colunas = {
    'Renda média domiciliar': ['renda média', 'renda media'],
    'População Mulheres': ['mulheres'],
    'População Homens': ['homens'],
    'PEA Dia': ['pea'],
    '    de 20 a 24 anos': ['20 a 24'],
    'População': ['população'],
    'Densidade demográfica': ['densidade']
}

# Métricas dos modelos (valores que você forneceu)
METRICAS_MODELO = {
    'r2': 0.8944,
    'rmse': 200.09,
    'mae': 140.46
}

METRICAS_AGREGADORES = {
    'r2': 0.8621,  # Ajuste conforme suas métricas reais
    'rmse': 263.95,
    'mae': 156.81
}

def criar_grafico_comparativo(previsao_recorrentes, previsao_agregadores, metricas_recorrentes, metricas_agregadores):
    """Cria gráfico de barras comparando ambos os modelos"""
    # Criar figura com fundo branco
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Dados para o gráfico de alunos recorrentes
    categorias_recorrentes = ['Mínimo', 'Previsão', 'Máximo']
    valores_recorrentes = [metricas_recorrentes['min_previsto'], previsao_recorrentes, metricas_recorrentes['max_previsto']]
    
    # Dados para o gráfico de agregadores
    categorias_agregadores = ['Mínimo', 'Previsão', 'Máximo']
    valores_agregadores = [metricas_agregadores['min_previsto'], previsao_agregadores, metricas_agregadores['max_previsto']]
    
    # Cores harmoniosas
    cores_recorrentes = ['#FF6B6B', '#EE5A24', '#00B894']
    cores_agregadores = ['#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Gráfico de alunos recorrentes
    barras1 = ax1.bar(categorias_recorrentes, valores_recorrentes, color=cores_recorrentes, alpha=0.8)
    ax1.set_title('Alunos Recorrentes', fontsize=14, fontweight='bold', color='#2c3e50')
    ax1.set_ylabel('Quantidade de Alunos', fontsize=11, fontweight='bold')
    
    # Gráfico de agregadores
    barras2 = ax2.bar(categorias_agregadores, valores_agregadores, color=cores_agregadores, alpha=0.8)
    ax2.set_title('Alunos Agregadores', fontsize=14, fontweight='bold', color='#2c3e50')
    ax2.set_ylabel('Quantidade de Alunos', fontsize=11, fontweight='bold')
    
    # Adicionar valores nas barras - Recorrentes
    for i, (barra, valor) in enumerate(zip(barras1, valores_recorrentes)):
        ax1.text(barra.get_x() + barra.get_width()/2, 
                barra.get_height() + max(valores_recorrentes)*0.02,
                f'{valor:.0f}', 
                ha='center', 
                va='bottom', 
                fontweight='bold', 
                fontsize=12,
                color=cores_recorrentes[i])
    
    # Adicionar valores nas barras - Agregadores
    for i, (barra, valor) in enumerate(zip(barras2, valores_agregadores)):
        ax2.text(barra.get_x() + barra.get_width()/2, 
                barra.get_height() + max(valores_agregadores)*0.02,
                f'{valor:.0f}', 
                ha='center', 
                va='bottom', 
                fontweight='bold', 
                fontsize=12,
                color=cores_agregadores[i])
    
    # Configurações comuns
    for ax in [ax1, ax2]:
        # Remover bordas
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Grid sutil
        ax.grid(True, alpha=0.2, axis='y', color='#bdc3c7')
        ax.grid(False, axis='x')
        
        # Fundo branco
        ax.set_facecolor('white')
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    
    # Salvar em memória
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

def calcular_metricas(previsao, tipo_modelo='recorrentes'):
    """Calcula métricas baseadas na previsão e nas métricas do modelo"""
    
    if tipo_modelo == 'recorrentes':
        metricas_base = METRICAS_MODELO
    else:  # agregadores
        metricas_base = METRICAS_AGREGADORES
    
    # Calcular intervalo de confiança baseado no RMSE
    margem_erro = metricas_base['rmse'] * 1.645
    
    metricas = {
        'previsao': previsao,
        'min_previsto': max(0, previsao - margem_erro),
        'max_previsto': previsao + margem_erro,
        'intervalo_confianca': f"{max(0, previsao - margem_erro):.0f} - {previsao + margem_erro:.0f}",
        'margem_erro': margem_erro,
        'r2': metricas_base['r2'],
        'rmse': metricas_base['rmse'],
        'mae': metricas_base['mae'],
        'acuracia_percentual': f"{(metricas_base['r2'] * 100):.1f}%"
    }
    
    return metricas

def preparar_dados_entrada(df, features_necessarias, vagas, metragem, concorrentes):
    """Prepara os dados de entrada para qualquer modelo"""
    valores = []
    for feature in features_necessarias:
        if feature in colunas_manuais:
            # Valores manuais
            if feature == 'Vagas ':
                valores.append(vagas)
            elif feature == 'Metragem':
                valores.append(metragem)
            elif feature == 'Quantidade de Concorrentes':
                valores.append(concorrentes)
        else:
            # Buscar na planilha
            if df is not None:
                palavras = mapa_colunas.get(feature, [])
                coluna_encontrada = None
                for col in df.columns:
                    if any(p.lower() in col.lower() for p in palavras):
                        coluna_encontrada = col
                        break
                valor_planilha = df[coluna_encontrada].iloc[0] if coluna_encontrada else 0
                valores.append(float(valor_planilha))
            else:
                valores.append(0)  # Valor padrão se não tem planilha
    
    return valores

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prever', methods=['POST'])
def prever():
    try:
        # Receber dados do formulário
        vagas = float(request.form['vagas'])
        metragem = float(request.form['metragem'])
        concorrentes = float(request.form['concorrentes'])
        
        # Processar arquivo Excel se foi enviado
        planilha = request.files.get('planilha')
        df = None
        if planilha and planilha.filename.endswith('.xlsx'):
            df = pd.read_excel(planilha)
        
        # Fazer previsão para alunos recorrentes
        valores_recorrentes = preparar_dados_entrada(df, features, vagas, metragem, concorrentes)
        dados_recorrentes = pd.DataFrame([valores_recorrentes], columns=features)
        previsao_recorrentes = modelo.predict(dados_recorrentes)[0]
        
        # Fazer previsão para agregadores
        valores_agregadores = preparar_dados_entrada(df, features_agregadores, vagas, metragem, concorrentes)
        dados_agregadores = pd.DataFrame([valores_agregadores], columns=features_agregadores)
        previsao_agregadores = modelo_agregadores.predict(dados_agregadores)[0]
        
        # Calcular métricas para ambos
        metricas_recorrentes = calcular_metricas(previsao_recorrentes, 'recorrentes')
        metricas_agregadores = calcular_metricas(previsao_agregadores, 'agregadores')
        
        # Criar gráfico comparativo
        grafico_base64 = criar_grafico_comparativo(
            previsao_recorrentes, 
            previsao_agregadores, 
            metricas_recorrentes, 
            metricas_agregadores
        )
        
        # Calcular total geral
        total_geral = previsao_recorrentes + previsao_agregadores
        
        return jsonify({
            'previsao_recorrentes': f'{previsao_recorrentes:.0f}',
            'previsao_agregadores': f'{previsao_agregadores:.0f}',
            'total_geral': f'{total_geral:.0f}',
            'status': 'success',
            'mensagem': f'Previsões calculadas com sucesso!',
            'metricas_recorrentes': metricas_recorrentes,
            'metricas_agregadores': metricas_agregadores,
            'grafico': grafico_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)