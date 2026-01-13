# 🔍 Machine Learning Project - Turbine Fault Detection

[![Python](https://img.shields.io/badge/Python-3.13.3-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn%20%7C%20XGBoost-FF6F00)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Sistema de detección de fallas en turbinas de gas utilizando algoritmos de Machine Learning para predicción y clasificación de estados operativos.

## 📋 Tabla de Contenidos
- [🎯 Descripción del Proyecto](#-descripción-del-proyecto)
- [⚙️ Problemas a Resolver](#️-problemas-a-resolver)
- [📊 Dataset](#-dataset)

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema de detección y clasificación de fallas en turbinas de gas mediante técnicas avanzadas de Machine Learning. El sistema monitorea el estado operativo de las turbinas y predice posibles fallos antes de que ocurran.

**Características principales:**
- 🔍 **Detección binaria**: Identifica si una turbina está en estado de falla o no
- 🎯 **Clasificación multiclase**: Determina el tipo específico de falla
- 📊 **Análisis predictivo**: Predice fallos basándose en patrones de degradación
- 🔄 **Simulación realista**: Datos generados a partir de ciclos de vida completos

## ⚙️ Problemas a Resolver

### **Objetivo Principal** - Detección Binaria
- **Clase 0**: ✅ No Falla (Estado normal de operación)
- **Clase 1**: ⚠️ Falla (Estado de falla detectado)

### **Objetivo Secundario** - Clasificación de Modos de Falla
- **1. 🔩 MECANICA_COJINETES**: Fallas en el sistema mecánico de cojinetes
- **2. 🌡️ ENFRIAMIENTO_PRESION**: Problemas en el sistema de enfriamiento y presión
- **3. ⚙️ CONTROL_COMBUSTIBLE**: Fallas en el sistema de control de combustible

## 📊 Dataset

### **📈 Características del Dataset Simulado**
| Característica | Valor |
|----------------|-------|
| **Turbinas simuladas** | 1,000 unidades |
| **Observaciones totales** | 100,000 registros |
| **Tipo de variables** | Todas numéricas |
| **Periodo simulado** | Ciclo de vida completo |
| **Degradación** | Progresiva hasta falla |

### **🔗 Dataset Original**
- **Nombre**: Gas Turbine Engine Fault Detection Dataset
- **Plataforma**: [Kaggle](https://www.kaggle.com/datasets/ziya07/gas-turbine-engine-fault-detection-dataset)
- **Propósito**: Base para la generación de datos simulados realistas


