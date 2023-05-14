import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos de Amazon Product Reviews
data = pd.read_csv('C:/Users/oem/Downloads/1429_1.csv/1429_1.csv')  # Reemplaza 'ruta_del_archivo.csv' por la ubicación real del archivo

# Eliminar filas con valores faltantes en la columna 'rating'
data = data.dropna(subset=['rating'])

# Preprocesamiento de datos: seleccionar características relevantes
features = ['name', 'brand', 'categories', 'manufacturer', 'rating']
data = data[features]

# Convertir características categóricas en variables dummy
data = pd.get_dummies(data)

# Dividir los datos en conjunto de entrenamiento y prueba
X = data.drop('rating', axis=1)
y = data['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de clasificación (Árbol de decisión)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy}')

# Obtener el perfil del producto desde la entrada de texto del usuario
name = input('Nombre del producto: ')
brand = input('Marca del producto: ')
categories = input('Categorías del producto: ')
manufacturer = input('Fabricante del producto: ')

# Crear el perfil del usuario a partir de la entrada de texto
user_profile = {
    'name': name,
    'brand': brand,
    'categories': categories,
    'manufacturer': manufacturer
}

# Convertir el perfil de usuario en un dataframe
user_profile_df = pd.DataFrame(user_profile, index=[0])

# Convertir características categóricas en variables dummy
user_profile_df = pd.get_dummies(user_profile_df)

# Asegurarse de que las características del perfil del usuario coincidan con las características utilizadas durante el entrenamiento
missing_cols = set(X.columns) - set(user_profile_df.columns)
for col in missing_cols:
    user_profile_df[col] = 0

# Reordenar las columnas para que coincidan con el orden utilizado durante el entrenamiento
user_profile_df = user_profile_df[X.columns]

# Calificación predicha para el usuario
predicted_rating = model.predict(user_profile_df)

print(f'Calificación predicha para el usuario: {predicted_rating}')

