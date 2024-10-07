import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

# Cargar el archivo CSV
data = pd.read_csv("ruta_al_archivo.csv")

# Supongamos que las columnas son 'Tiempo (s)' y 'Velocidad (m/s)'
tiempo = data['Tiempo (s)'].values
velocidad = data['Velocidad (m/s)'].values

# Escalar los datos de velocidad
scaler = MinMaxScaler()
velocidad = scaler.fit_transform(velocidad.reshape(-1, 1))

time_steps = 10  

indice_inicio_onda = 100  

etiquetas = np.zeros(len(velocidad))
etiquetas[indice_inicio_onda:] = 1  

def crear_secuencias(velocidad, etiquetas, time_steps):
    X, y = [], []
    for i in range(len(velocidad) - time_steps):
        X.append(velocidad[i:i+time_steps])
        y.append(etiquetas[i+time_steps])  
    return np.array(X), np.array(y)

X, y = crear_secuencias(velocidad, etiquetas, time_steps)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Salida binaria (inicio de onda: 1 o no: 0)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

model.save("LSTM.h5")
predicciones = model.predict(X_test)

for i in range(5):
    print(f"Predicción: {predicciones[i]}, Valor real: {y_test[i]}")

nuevo_data = pd.read_csv("ruta_nuevo_archivo.csv")

nuevo_tiempo = nuevo_data['Tiempo (s)'].values
nuevo_velocidad = nuevo_data['Velocidad (m/s)'].values
nuevo_velocidad = scaler.transform(nuevo_velocidad.reshape(-1, 1))

nuevo_X, _ = crear_secuencias(nuevo_velocidad, np.zeros(len(nuevo_velocidad)), time_steps)

nuevo_predicciones = model.predict(nuevo_X)

umbral = 0.5
indice_inicio_nueva_onda = np.where(nuevo_predicciones > umbral)[0][0]

print(f"La nueva onda comienza en el tiempo {nuevo_tiempo[indice_inicio_nueva_onda + time_steps]:.2f} segundos")
