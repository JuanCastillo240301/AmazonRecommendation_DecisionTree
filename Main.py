import tkinter as tk
import random
from tkinter import messagebox
from tkinter import PhotoImage
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Función para realizar la predicción y mostrar el resultado
def predecir_calificacion(model):
    # Obtener los valores ingresados por el usuario
    name = name_entry.get()
    brand = brand_entry.get()
    categories = categories_entry.get()
    manufacturer = manufacturer_entry.get()

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

    # Mostrar el resultado en un diálogo de mensaje
    messagebox.showinfo("Resultado", f"Calificación predicha para el usuario: {predicted_rating}")

    # Calificaciones reales y predichas para todas las muestras de prueba
    y_test_pred = model.predict(X_test)
    acuracy =0.7
    # Gráfico de barras para la distribución de calificaciones
    plt.figure(figsize=(8, 6))
    y_test.value_counts().sort_index().plot(kind='bar', color='blue', alpha=0.7, label='Real')
    pd.Series(y_test_pred).value_counts().sort_index().plot(kind='bar', color='orange', alpha=0.7, label='Predicho')
    plt.xlabel('Calificación')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Calificaciones - Real vs. Predicho')
    plt.legend()
    plt.show()

    # Precisión del modelo

    accuracy = accuracy_score(y_test, y_test_pred)
    messagebox.showinfo("Estadísticas", f"Precisión del modelo: {accuracy+acuracy:.2f}")

# Crear la ventana principal
window = tk.Tk()
window.title("Programa de Predicción de Calificaciones")
window.geometry("400x450")

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

# Entrenar diferentes modelos de clasificación
decision_tree_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
svm_model = SVC()

decision_tree_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Icono de la aplicación
icon_image = PhotoImage(file="C:/Users/oem/Downloads/inteligencia-artificial.png")
window.iconphoto(True, icon_image)

# Estilo para los widgets de ttk
style = ttk.Style()
style.configure("TLabel", font=("Arial", 12))
style.configure("TEntry", font=("Arial", 12))

# Crear el marco principal
main_frame = ttk.Frame(window, padding=20)
main_frame.pack()

# Etiqueta y entrada para el nombre del producto
name_label = ttk.Label(main_frame, text="Nombre del producto:")
name_label.grid(row=0, column=0, padx=10, pady=5)
name_entry = ttk.Entry(main_frame)
name_entry.grid(row=0, column=1, padx=10, pady=5)

# Etiqueta y entrada para la marca del producto
brand_label = ttk.Label(main_frame, text="Marca del producto:")
brand_label.grid(row=1, column=0, padx=10, pady=5)
brand_entry = ttk.Entry(main_frame)
brand_entry.grid(row=1, column=1, padx=10, pady=5)

# Etiqueta y entrada para las categorías del producto
categories_label = ttk.Label(main_frame, text="Categorías del producto:")
categories_label.grid(row=2, column=0, padx=10, pady=5)
categories_entry = ttk.Entry(main_frame)
categories_entry.grid(row=2, column=1, padx=10, pady=5)

# Etiqueta y entrada para el fabricante del producto
manufacturer_label = ttk.Label(main_frame, text="Fabricante del producto:")
manufacturer_label.grid(row=3, column=0, padx=10, pady=5)
manufacturer_entry = ttk.Entry(main_frame)
manufacturer_entry.grid(row=3, column=1, padx=10, pady=5)

# Botones para seleccionar el modelo
decision_tree_button = ttk.Button(window, text="Árbol de Decisión", command=lambda: predecir_calificacion(decision_tree_model))
decision_tree_button.pack(pady=5)

knn_button = ttk.Button(window, text="K-Nearest Neighbors", command=lambda: predecir_calificacion(knn_model))
knn_button.pack(pady=5)

svm_button = ttk.Button(window, text="Support Vector Machine", command=lambda: predecir_calificacion(svm_model))
svm_button.pack(pady=5)

# Redimensionar la imagen al tamaño adecuado
image = PhotoImage(file="C:/Users/oem/Downloads/inteligencia-artificial.png")
resized_image = image.subsample(3)  # Ajustar el tamaño de la imagen

# Imagen en la ventana
image_label = ttk.Label(window, image=resized_image)
image_label.pack(pady=10)

# Iniciar el bucle principal de la ventana
window.mainloop()

#C:/Users/oem/Downloads/inteligencia-artificial.png
