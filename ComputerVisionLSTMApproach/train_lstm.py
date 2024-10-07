import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

# Configuración para utilizar GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Cargar los índices de inicio de las ondas desde numeros.csv
indices_inicios = pd.read_csv("numeros.csv", header=None).values.flatten()  # Asume que hay una columna sin encabezado

# Inicializa listas para los datos combinados
todos_tiempos = []
todos_velocidades = []
etiquetas = []

# Iterar sobre los archivos de datos
for i in range(1,len(indices_inicios)):  # Asume que hay una cantidad de archivos CSV igual al número de índices
    archivo_csv = f"deteciones_row_{i}.csv"  # Cambia esto según tu nomenclatura de archivos
    try:
        print(f"Intentando cargar el archivo: {archivo_csv}")  # Imprimir el archivo que se está intentando cargar
        data = pd.read_csv(archivo_csv)

        tiempo = data['Tiempo (s)'].values
        velocidad = data['Velocidad (m/s)'].values

        # Escalar las velocidades
        scaler = MinMaxScaler()
        velocidad = scaler.fit_transform(velocidad.reshape(-1, 1))

        # Agregar datos a las listas
        todos_tiempos.extend(tiempo)
        todos_velocidades.extend(velocidad.flatten())  # Aplanar el arreglo

        # Etiquetar según el índice de inicio de la onda
        etiqueta = np.zeros(len(velocidad))

        # Convertir a entero y verificar el rango
        indice_inicio = int(indices_inicios[i-1])  # Asegurarte de que sea un entero
        print(f"Índice de inicio para el archivo {i-1}: {indice_inicio}")  # Imprimir el índice de inicio

        if 0 <= indice_inicio < len(etiqueta):  # Verifica que esté dentro de los límites
            print(f"Marcando la etiqueta desde el índice {indice_inicio} hasta el final.")  # Imprimir el rango que se marcará
            etiqueta[indice_inicio:] = 1
        else:
            print(f"Índice de inicio {indice_inicio} fuera de rango para el archivo {archivo_csv}.")  # Avisar si el índice está fuera de rango

        etiquetas.extend(etiqueta)

    except FileNotFoundError:
        print(f"Archivo no encontrado: {archivo_csv}. Se salta este archivo.")
        continue  # Pasar al siguiente archivo si no se encuentra
    except Exception as e:
        print(f"Ocurrió un error al procesar el archivo {archivo_csv}: {e}")  # Imprimir cualquier otro error





# Convertir a arrays de numpy
todos_tiempos = np.array(todos_tiempos)
todos_velocidades = np.array(todos_velocidades).reshape(-1, 1)
etiquetas = np.array(etiquetas)

# Escalar los datos de velocidad
scaler = MinMaxScaler()
todos_velocidades = scaler.fit_transform(todos_velocidades)

# Definir el número de pasos de tiempo
time_steps = 10  

# Crear las secuencias
def crear_secuencias(velocidad, etiquetas, time_steps):
    X, y = [], []
    for i in range(len(velocidad) - time_steps):
        X.append(velocidad[i:i + time_steps])
        y.append(etiquetas[i + time_steps])
    return np.array(X), np.array(y)

X, y = crear_secuencias(todos_velocidades, etiquetas, time_steps)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

# Guardar el modelo
model.save("LSTM.h5")

# Realizar predicciones
predicciones = model.predict(X_test)
for i in range(5):
    print(f"Predicción: {predicciones[i]}, Valor real: {y_test[i]}")
