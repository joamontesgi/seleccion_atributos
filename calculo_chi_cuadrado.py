from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2

# Cargar el dataset
iris = load_iris()
X, y = iris.data, iris.target

# k = int(input("Ingrese el número de atributos: "))

# Seleccionando las 3 mejores características
selector = SelectKBest(chi2, k=1)
x_new = selector.fit_transform(X, y)

# Obtener las características seleccionadas
features_selected = selector.get_support(indices=True)

# Usando listas por comprensión para iterar sobre las columnas del dataset y las columnas obtenidas
selected_features_names = [iris.feature_names[i] for i in features_selected]

# Mostrar las características obtenidas
for feature in features_selected:
    print("-",feature)
    
print("La cantidad de características  seleccionadas: ", len(selected_features_names))
