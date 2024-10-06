import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

# Ruta al archivo CSV
cat_file = r"D:\Space Aps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar\training\catalogs\apollo12_catalog_GradeA_final.csv"
data_folder = r"C:\Users\Luciano\Downloads\Spectogram0.0.v1i.yolov8\CSV_LTSM"

def load_and_preprocess_data(data_folder, cat_file):
    data_list = []
    
    # Cargar el catálogo
    cat = pd.read_csv(cat_file, skiprows=[21, 43, 55])
    
    # Inicializar un escalador
    scaler = MinMaxScaler()
    label_encoder = LabelEncoder()  # Para convertir las etiquetas a numéricas

    for idx in range(len(cat)):
        filename = f"deteciones_row_{idx + 1}.csv"
        label = cat.iloc[idx, 4]  # Obtener la etiqueta de la quinta columna
        filepath = os.path.join(data_folder, filename)
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                
                # Normalizar los datos
                df['Velocidad (m/s)'] = scaler.fit_transform(df[['Velocidad (m/s)']])
                
                # Guardar datos y etiqueta en un diccionario
                data_list.append({'data': df[['Tiempo (s)', 'Velocidad (m/s)']].values, 'label': label})
            except Exception as e:
                print(f"Error al cargar el archivo {filename}: {e}")
        else:
            print(f"Archivo no encontrado: {filepath}")

    # Convertir las etiquetas a numéricas
    labels = [entry['label'] for entry in data_list]
    encoded_labels = label_encoder.fit_transform(labels)  # Convertir etiquetas a numéricas
    
    for idx, entry in enumerate(data_list):
        entry['label'] = encoded_labels[idx]  # Asignar etiqueta codificada
    
    return data_list

def prepare_data(data_list, time_steps=10):
    X, y = [], []
    
    for entry in data_list:
        data = entry['data']  # Datos de tiempo y velocidad
        label = entry['label']  # Etiqueta
        
        # Crear secuencias
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])  # Secuencia de datos
            y.append(label)  # Etiqueta correspondiente
        
    return np.array(X), np.array(y)

# Cargar y preprocesar los datos
data_list = load_and_preprocess_data(data_folder, cat_file)

# Preparar los datos para LSTM
time_steps = 10  # Número de pasos temporales
X, y = prepare_data(data_list, time_steps)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Regularización para evitar overfitting
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))  # Salida (ajustar según tu caso)

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')  # Usa 'categorical_crossentropy' si es clasificación

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')

model.save("LSTM Trained.h5")

# Hacer predicciones
predictions = model.predict(X_test)

# Opcional: Imprimir las primeras 5 predicciones
print(predictions[:5])





"""
def load_and_preprocess_data(data_folder, cat_file):
    data = []
    labels = []
    
    # Cargar el catálogo omitiendo líneas problemáticas
    cat = pd.read_csv(cat_file, skiprows=[21, 43, 55])
    
    # Inicializar un escalador
    scaler = MinMaxScaler()

    for idx in range(len(cat)):
        filename = f"deteciones_row_{idx + 1}.csv"
        label = cat.iloc[idx, 4]  # Obtener la etiqueta de la quinta columna
        filepath = os.path.join(data_folder, filename)
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                
                # Normalizar los datos
                df['Velocidad (m/s)'] = scaler.fit_transform(df[['Velocidad (m/s)']])
                
                # Agregar todos los datos y etiquetas a las listas
                data.append(df[['Tiempo (s)', 'Velocidad (m/s)']].values)  # Guardar todos los valores de tiempo y velocidad
                labels.append(label)  # Guardar la etiqueta correspondiente
            except Exception as e:
                print(f"Error al cargar el archivo {filename}: {e}")
        else:
            print(f"Archivo no encontrado: {filepath}")

    # Convertir listas a arrays de numpy
    data = np.concatenate(data) if data else np.array([])  # Concatenar todos los arrays de datos
    labels = np.array(labels)

    return data, labels


# Cargar los datos
data, labels = load_and_preprocess_data(data_folder, cat_file)

print(data)
print(labels)
print("Data length", len(data))
print("labels", len(labels))

"""
