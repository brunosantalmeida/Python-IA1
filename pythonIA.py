from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Iris
dados = load_iris()
X = dados.data
y = dados.target

# Dividir o conjunto de dados em treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo de floresta aleatória
modelo = RandomForestClassifier()
modelo.fit(X_treino, y_treino)

# Fazer previsões no conjunto de teste
y_previsto = modelo.predict(X_teste)

# Avaliar a precisão do modelo
precisao = accuracy_score(y_teste, y_previsto)
print(f'Precisão do modelo: {precisao * 100:.2f}%')
