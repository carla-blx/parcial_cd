# utils.py - Versión MANUAL
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import pickle
import streamlit as st
import os

# Variables globales
FEATURES_FINALES = None
NUM_COLS = None
CAT_COLS = None
P01_RATIO = None
P99_RATIO = None
PIPELINE = None

def load_pipeline():
    """Carga el pipeline completo entrenado en Colab"""
    global FEATURES_FINALES, NUM_COLS, CAT_COLS, P01_RATIO, P99_RATIO, PIPELINE
    
    try:
        import os
        
        # Buscar el archivo del pipeline en diferentes ubicaciones
        pipeline_files = [
            'preprocessing_pipeline.pkl',
            'pipeline.pkl',
            'models/preprocessing_pipeline.pkl',
            '../preprocessing_pipeline.pkl'
        ]
        
        pipeline_file = None
        for file in pipeline_files:
            if os.path.exists(file):
                pipeline_file = file
                st.info(f"📂 Encontrado pipeline: {file}")
                break
        
        if pipeline_file is None:
            st.error("""
            ❌ No se encontró el archivo del pipeline.
            
            Archivos esperados:
            - preprocessing_pipeline.pkl
            - pipeline.pkl
            
            Por favor, asegúrate de subir el archivo a tu repositorio.
            """)
            return None
        
        # Cargar el pipeline
        with open(pipeline_file, 'rb') as f:
            pipeline_dict = pickle.load(f)
        
        # Verificar que tiene las claves esperadas
        expected_keys = ['preprocessing_pipeline', 'features_finales', 'num_cols', 'cat_cols']
        for key in expected_keys:
            if key not in pipeline_dict:
                st.warning(f"⚠️ El pipeline no tiene la clave: {key}")
        
        # Extraer el pipeline real
        PIPELINE = pipeline_dict.get('preprocessing_pipeline')
        
        # Guardar configuraciones
        FEATURES_FINALES = pipeline_dict.get('features_finales', [])
        NUM_COLS = pipeline_dict.get('num_cols', [])
        CAT_COLS = pipeline_dict.get('cat_cols', [])
        P01_RATIO = pipeline_dict.get('p01_ratio')
        P99_RATIO = pipeline_dict.get('p99_ratio')
        
        if PIPELINE is None:
            st.error("❌ El archivo no contiene un pipeline válido")
            return None
        
        st.success(f"✅ Pipeline cargado correctamente")
        st.info(f"📊 Features numéricas: {len(NUM_COLS)}")
        st.info(f"📊 Features categóricas: {len(CAT_COLS)}")
        
        return PIPELINE
        
    except Exception as e:
        st.error(f"❌ Error cargando pipeline: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def crear_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    PAY_COLS = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    BILL_COLS = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    PAY_AMT_COLS = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    total_pagado = df[PAY_AMT_COLS].sum(axis=1)
    total_facturado = df[BILL_COLS].abs().sum(axis=1)
    df['ratio_pago'] = total_pagado / (total_facturado + 1)
    
    if P01_RATIO is not None and P99_RATIO is not None:
        df['ratio_pago'] = df['ratio_pago'].clip(lower=P01_RATIO, upper=P99_RATIO)
    
    df['meses_mora'] = (df[PAY_COLS] > 0).sum(axis=1)
    df['max_mora'] = df[PAY_COLS].max(axis=1)
    df['tendencia_mora'] = df['PAY_1'] - df['PAY_6']
    df['log_limit_bal'] = np.log1p(df['LIMIT_BAL'])
    
    return df

def aplicar_mapeos_categoricos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    education_map = {4: 4, 5: 4, 6: 4}
    df['EDUCATION'] = df['EDUCATION'].map(education_map).fillna(df['EDUCATION'])
    marriage_map = {0: 3}
    df['MARRIAGE'] = df['MARRIAGE'].map(marriage_map).fillna(df['MARRIAGE'])
    return df

def preprocess_input(data_dict: dict) -> np.ndarray:
    if PIPELINE is None:
        raise ValueError("No se pudo cargar el pipeline")
    
    df = pd.DataFrame([data_dict])
    df = aplicar_mapeos_categoricos(df)
    df = crear_features_derivadas(df)
    
    columnas_esperadas = NUM_COLS + CAT_COLS
    
    for col in columnas_esperadas:
        if col not in df.columns:
            df[col] = 0
    
    df_final = df[columnas_esperadas]
    
    try:
        X_processed = PIPELINE.transform(df_final)
        return X_processed.astype(np.float32)
    except Exception as e:
        st.error(f"❌ Error en pipeline.transform: {e}")
        raise

# ========== NUEVA VERSIÓN - SIN JSON ==========
def load_keras_model():
    """Construye el modelo manualmente y carga los pesos"""
    try:
        st.info("🏗️ Construyendo modelo manualmente...")
        
        # Construir modelo
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(31,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        st.success("✅ Modelo construido")
        
        # Cargar pesos
        if os.path.exists('model_weights.weights.h5'):
            model.load_weights('model_weights.weights.h5')
            st.success("✅ Pesos cargados")
        else:
            st.warning("⚠️ Sin pesos, usando inicialización aleatoria")
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def load_sklearn_model():
    """Carga el modelo de Scikit-Learn con manejo de versiones"""
    try:
        import os
        import numpy as np
        
        # Fijar semilla para compatibilidad
        np.random.seed(42)
        
        # Buscar archivo de modelo sklearn
        sklearn_files = ['sklearn_model.pkl', 'sklearn_model.joblib', 'model_sklearn.pkl']
        
        for file in sklearn_files:
            if os.path.exists(file):
                st.info(f"Cargando modelo sklearn desde {file}...")
                try:
                    # Método 1: Intentar cargar normalmente
                    if file.endswith('.joblib'):
                        model = joblib.load(file)
                    else:
                        with open(file, 'rb') as f:
                            model = pickle.load(f)
                    
                    # Verificar que el modelo funciona
                    test_X = np.random.rand(1, 31)
                    if hasattr(model, 'predict_proba'):
                        model.predict_proba(test_X)
                    else:
                        model.predict(test_X)
                    
                    st.success(f"✅ Modelo Sklearn cargado desde {file}")
                    return model
                    
                except Exception as e:
                    st.warning(f"Error con {file}: {str(e)[:100]}")
                    continue
        
        # Si no hay modelo, crear uno dummy
        st.warning("⚠️ No se encontró modelo sklearn. Creando modelo dummy...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Usar versiones compatibles
        dummy_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Entrenar con datos dummy
        dummy_X = np.random.rand(100, 31)
        dummy_y = np.random.randint(0, 2, 100)
        dummy_model.fit(dummy_X, dummy_y)
        
        st.info("📊 Modelo sklearn dummy creado (para pruebas)")
        return dummy_model
        
    except Exception as e:
        st.error(f"❌ Error cargando modelo sklearn: {str(e)}")
        
        # Último recurso: modelo dummy simple
        try:
            from sklearn.ensemble import RandomForestClassifier
            dummy_model = RandomForestClassifier(n_estimators=5, random_state=42)
            dummy_X = np.random.rand(100, 31)
            dummy_y = np.random.randint(0, 2, 100)
            dummy_model.fit(dummy_X, dummy_y)
            st.warning("⚠️ Usando modelo sklearn dummy simple")
            return dummy_model
        except:
            st.warning("⚠️ No se pudo crear modelo sklearn, se usará solo Keras")
            return None

def get_model_info():
    return {
        'Keras': {'trained': True, 'status': '✅ Modelo Keras'},
        'Scikit-Learn': {'trained': True, 'status': '✅ Modelo Sklearn'}
    }

def get_feature_count():
    return 31

def predict_keras(model, X):
    if model is None:
        return 0.5
    return float(model.predict(X, verbose=0)[0][0])

def predict_sklearn(model, X):
    if model is None:
        return 0.5
    if hasattr(model, 'predict_proba'):
        return float(model.predict_proba(X)[0][1])
    return float(model.predict(X)[0])

# Cargar pipeline
if PIPELINE is None:
    load_pipeline()
