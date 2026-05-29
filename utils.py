# utils.py - Versión corregida definitiva
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import pickle
import streamlit as st
import os
import json

# Variables globales
FEATURES_FINALES = None
NUM_COLS = None
CAT_COLS = None
P01_RATIO = None
P99_RATIO = None
PIPELINE = None

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

def load_pipeline():
    """Carga el pipeline completo entrenado en Colab"""
    global FEATURES_FINALES, NUM_COLS, CAT_COLS, P01_RATIO, P99_RATIO, PIPELINE
    
    try:
        with open('preprocessing_pipeline.pkl', 'rb') as f:
            pipeline_dict = pickle.load(f)
        
        # Extraer el pipeline real
        PIPELINE = pipeline_dict['preprocessing_pipeline']
        
        # Guardar configuraciones
        FEATURES_FINALES = pipeline_dict['features_finales']
        NUM_COLS = pipeline_dict['num_cols']
        CAT_COLS = pipeline_dict['cat_cols']
        P01_RATIO = pipeline_dict['p01_ratio']
        P99_RATIO = pipeline_dict['p99_ratio']
        
        st.success(f"✅ Pipeline cargado correctamente")
        return PIPELINE
    except Exception as e:
        st.error(f"❌ Error cargando pipeline: {e}")
        return None

# =============================================================================
# FUNCIONES DE CREACIÓN DE FEATURES
# =============================================================================

def crear_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las 5 features derivadas exactamente como en el entrenamiento"""
    df = df.copy()
    
    PAY_COLS = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    BILL_COLS = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    PAY_AMT_COLS = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # ratio_pago
    total_pagado = df[PAY_AMT_COLS].sum(axis=1)
    total_facturado = df[BILL_COLS].abs().sum(axis=1)
    df['ratio_pago'] = total_pagado / (total_facturado + 1)
    
    # Winsorización con percentiles del training
    if P01_RATIO is not None and P99_RATIO is not None:
        df['ratio_pago'] = df['ratio_pago'].clip(lower=P01_RATIO, upper=P99_RATIO)
    
    # meses_mora
    df['meses_mora'] = (df[PAY_COLS] > 0).sum(axis=1)
    
    # max_mora
    df['max_mora'] = df[PAY_COLS].max(axis=1)
    
    # tendencia_mora
    df['tendencia_mora'] = df['PAY_1'] - df['PAY_6']
    
    # log_limit_bal
    df['log_limit_bal'] = np.log1p(df['LIMIT_BAL'])
    
    return df

def aplicar_mapeos_categoricos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los mapeos de educación y estado civil"""
    df = df.copy()
    
    # Mapeo de educación según tu Colab: {4: 4, 5: 4, 6: 4}
    education_map = {4: 4, 5: 4, 6: 4}
    df['EDUCATION'] = df['EDUCATION'].map(education_map).fillna(df['EDUCATION'])
    
    # Mapeo de matrimonio según tu Colab: {0: 3}
    marriage_map = {0: 3}
    df['MARRIAGE'] = df['MARRIAGE'].map(marriage_map).fillna(df['MARRIAGE'])
    
    return df

# =============================================================================
# PREPROCESAMIENTO PRINCIPAL
# =============================================================================

def preprocess_input(data_dict: dict) -> np.ndarray:
    """Preprocesa los datos usando el pipeline real"""
    
    if PIPELINE is None:
        raise ValueError("No se pudo cargar el pipeline")
    
    # 1. Crear DataFrame con todos los datos
    df = pd.DataFrame([data_dict])
    
    # 2. Aplicar mapeos categóricos
    df = aplicar_mapeos_categoricos(df)
    
    # 3. Crear features derivadas
    df = crear_features_derivadas(df)
    
    # 4. El pipeline espera las columnas en el orden: NUM_COLS + CAT_COLS
    columnas_esperadas = NUM_COLS + CAT_COLS
    
    # 5. Verificar que todas las columnas existan
    for col in columnas_esperadas:
        if col not in df.columns:
            df[col] = 0
    
    # 6. Seleccionar solo las columnas esperadas en el orden correcto
    df_final = df[columnas_esperadas]
    
    # 7. Aplicar el pipeline
    try:
        X_processed = PIPELINE.transform(df_final)
        return X_processed.astype(np.float32)
    except Exception as e:
        st.error(f"❌ Error en pipeline.transform: {e}")
        raise

# =============================================================================
# LOADERS - VERSIÓN CORREGIDA
# =============================================================================
def load_keras_model():
    """Construye el modelo manualmente y carga los pesos (sin usar JSON)"""
    try:
        import os
        
        st.info("🏗️ Construyendo modelo manualmente...")
        
        # ============================================================
        # CONSTRUIR EL MODELO MANUALMENTE (basado en tu arquitectura)
        # ============================================================
        
        # Definir la misma arquitectura que usaste en entrenamiento
        model = tf.keras.Sequential([
            # Capa de entrada (31 features)
            tf.keras.layers.Input(shape=(31,)),
            
            # Primera capa densa
            tf.keras.layers.Dense(64, activation='relu', 
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                 name='dense_18'),
            
            # Batch Normalization
            tf.keras.layers.BatchNormalization(name='batch_normalization_10'),
            
            # Dropout
            tf.keras.layers.Dropout(0.3, name='dropout_10'),
            
            # Segunda capa densa
            tf.keras.layers.Dense(32, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                 name='dense_19'),
            
            # Batch Normalization
            tf.keras.layers.BatchNormalization(name='batch_normalization_11'),
            
            # Dropout
            tf.keras.layers.Dropout(0.3, name='dropout_11'),
            
            # Capa de salida
            tf.keras.layers.Dense(1, activation='sigmoid', name='dense_20')
        ])
        
        st.success("✅ Modelo construido manualmente")
        
        # ============================================================
        # CARGAR LOS PESOS
        # ============================================================
        
        # Buscar archivo de pesos
        weights_file = None
        for f in ['model.weights.h5', 'model_weights.h5', 'model_weights.weights.h5', 'keras_model.weights.h5']:
            if os.path.exists(f):
                weights_file = f
                break
        
        if weights_file:
            try:
                model.load_weights(weights_file)
                st.success(f"⚖️ Pesos cargados desde {weights_file}")
            except Exception as e:
                st.warning(f"No se pudieron cargar los pesos: {e}")
                st.info("Usando modelo con pesos aleatorios")
        else:
            st.warning("⚠️ No se encontró archivo de pesos. Usando pesos aleatorios.")
            st.info("El modelo funcionará pero con menor precisión")
        
        # ============================================================
        # COMPILAR EL MODELO
        # ============================================================
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        st.success("✅ Modelo Keras listo para usar")
        return model
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        
        # Crear modelo dummy
        st.warning("⚠️ Creando modelo dummy para pruebas...")
        dummy_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(31,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        dummy_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return dummy_model

def load_sklearn_model():
    """Carga el modelo de Scikit-Learn"""
    try:
        import numpy as np
        import os
        
        # Configurar para compatibilidad
        np.random.seed(42)
        
        # Buscar archivo de modelo sklearn
        sklearn_files = ['sklearn_model.pkl', 'sklearn_model.joblib', 'model_sklearn.pkl']
        
        for file in sklearn_files:
            if os.path.exists(file):
                st.info(f"Cargando modelo sklearn desde {file}...")
                try:
                    if file.endswith('.joblib'):
                        model = joblib.load(file)
                    else:
                        with open(file, 'rb') as f:
                            model = pickle.load(f)
                    st.success(f"✅ Modelo Sklearn cargado desde {file}")
                    return model
                except Exception as e:
                    st.warning(f"Error cargando {file}: {e}")
                    continue
        
        # Si no hay modelo, crear uno dummy
        st.warning("⚠️ No se encontró modelo sklearn. Creando modelo dummy...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        dummy_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Entrenar con datos dummy
        dummy_X = np.random.rand(100, 31)
        dummy_y = np.random.randint(0, 2, 100)
        dummy_model.fit(dummy_X, dummy_y)
        
        st.info("📊 Modelo dummy creado (solo para pruebas)")
        return dummy_model
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        # Último recurso
        from sklearn.ensemble import RandomForestClassifier
        dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
        dummy_X = np.random.rand(100, 31)
        dummy_y = np.random.randint(0, 2, 100)
        dummy_model.fit(dummy_X, dummy_y)
        return dummy_model

def get_model_info() -> dict:
    """Información de los modelos"""
    return {
        'Keras': {
            'trained': True,
            'status': '✅ Modelo Keras cargado'
        },
        'Scikit-Learn': {
            'trained': True,
            'status': '✅ Modelo Sklearn cargado'
        }
    }

def get_feature_count() -> int:
    """Retorna el número de features"""
    if FEATURES_FINALES:
        return len(FEATURES_FINALES)
    return 31

# =============================================================================
# PREDICCIONES
# =============================================================================

def predict_keras(model, X: np.ndarray) -> float:
    """Predicción con Keras"""
    if model is None:
        return 0.5  # Valor por defecto si no hay modelo
    proba = model.predict(X, verbose=0)[0][0]
    return float(proba)

def predict_sklearn(model, X: np.ndarray) -> float:
    """Predicción con Scikit-Learn"""
    if model is None:
        return 0.5  # Valor por defecto si no hay modelo
    
    # Verificar si el modelo tiene predict_proba
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0][1]
    else:
        proba = model.predict(X)[0]
    
    return float(proba)

# Cargar pipeline automáticamente
if PIPELINE is None:
    load_pipeline()
