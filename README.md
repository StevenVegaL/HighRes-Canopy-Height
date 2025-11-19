# 🌳 High-Resolution Canopy Height  


<p align="center">
  <img src="assets/banner_chm.png" width="90%" />
</p>




### Proyecto final – Modelos Transformer aplicados a Imágenes

> Implementación inspirada en el artículo  
> **“High-resolution canopy height maps by learning from airborne lidar and spaceborne GEDI”**  
> Repositorio original: https://github.com/facebookresearch/HighResCanopyHeight

---

## 📌 Contexto y objetivo

El monitoreo forestal moderno necesita ir más allá del “bosque / no bosque” y aproximarse a un **censo estructural del bosque**:  
- ¿Cuánta área tiene árboles?  
- ¿Qué tan altos son esos árboles?  
- ¿Cómo se distribuye la altura del dosel en el territorio?

El artículo base propone un modelo capaz de **convertir imágenes satelitales RGB de muy alta resolución en mapas continuos de altura de dosel (~1 m)**, combinando información de:

- **LiDAR aéreo (ALS)** → detalle fino, pero cobertura limitada.
- **LiDAR satelital GEDI** → cobertura casi global, pero resolución (~25 m) y muestreo discreto (huellas).

Este proyecto reproduce y adapta ese enfoque usando **Transformers de visión pre-entrenados**, y construye una interfaz en **Streamlit** para explorar:

- Métricas científicas (MAE, RMSE, R², sesgo, altura P95).
- Métricas de **“censo estructural”** (altura promedio del dosel, % de área con árboles, distribución de alturas, etc.).

---

## 🧠 Descripción del modelo e innovaciones principales

El corazón del artículo (y de este repo) es un pipeline en varias fases:

1. **Pre-entrenamiento auto-supervisado (SSL) en imágenes satelitales**
   - Se usa un **ViT Huge** (Vision Transformer) pre-entrenado con **DINOv2** sobre **18 millones de imágenes satelitales Maxar**.
   - El modelo aprende a “entender” texturas de bosque, bordes de copas, sombras, caminos, etc. **sin etiquetas de altura**.
   - Resultado: un **encoder especializado en vegetación y paisaje**, que luego se reutiliza como backbone.

2. **Decoder DPT para altura de dosel de alta resolución (ALS)**
   - Encima del encoder congelado se entrena un **decoder DPT (Dense Prediction Transformer)**.
   - Entrada: imágenes RGB de sitios NEON (1 m GSD).  
   - Salida: mapas de altura de dosel (CHM) a la misma resolución.
   - Se utiliza:
     - **Arquitectura multi-escala (Reassemble + Fusion blocks)** para combinar contexto global y detalle fino.
     - **Pérdida Sigloss (tipo profundidad)** y **salida por bins (256 contenedores de altura)** para mejorar la estabilidad y evitar sesgos hacia alturas pequeñas.

3. **Modelo GEDI global (CNN + metadata)**
   - Se entrena un modelo separado (CNN) que:
     - Recibe parches RGB de 128×128.
     - Usa metadatos del haz GEDI: latitud, longitud, elevación solar, ángulo off-nadir y pendiente del terreno.
   - Predice la altura **RH95** (percentil 95 de altura) en el footprint de GEDI.
   - Esto permite tener un modelo consistente con las mediciones **globales** de GEDI, aunque sean de baja resolución.

4. **Fusión ALS + GEDI: mapa ajustado a escala global**
   - El mapa de CHM de alta resolución que se obtuvo con ALS se corrige usando el modelo GEDI:
     - El modelo GEDI actúa como una referencia global de “escala” de altura.
     - Se calcula un **factor de reescalamiento espacialmente variable** que ajusta el CHM ALS hacia la escala de GEDI.
   - Resultado: un **mapa continuo de altura de dosel**, con detalle de ~1 m, pero coherente con las alturas observadas por GEDI a escala global.

🔍 **Innovaciones clave:**

- Uso de **Transformers de visión pre-entrenados auto-supervisados** específicamente sobre **imágenes satelitales**, no solo datos genéricos tipo ImageNet.
- Arquitectura **DPT multi-escala** adaptada a mapas de altura de dosel:
  - Combina vista global del bosque y detalle de copas individuales.
- **Salida por bins + Sigloss**:
  - El modelo no predice directamente un escalar, sino una distribución discreta de alturas que luego se convierte en altura esperada.
  - Mejora estabilidad y reduce sesgos.
- **Fusión ALS + GEDI** para lograr:
  - Detalle local (ALS) + coherencia global (GEDI) en un solo CHM continuo.

---

## 🏗️ Resumen teórico de la arquitectura

Aquí se resume la arquitectura completa en 3 niveles: **encoder SSL**, **decoder ALS** y **modelo GEDI + fusión**.

---

### 1. Encoder SSL: ViT Huge con DINOv2

1. **Entrada**
   - Imágenes satelitales globales de 256×256 píxeles.
   - Se genera un **multi-crop**:
     - 2 vistas **globales**.
     - 8 vistas **locales** (algunas con máscara).

2. **Tokenización**
   - Cada imagen se divide en parches 16×16 → se aplanan a vectores.
   - Se proyectan a un embedding de dimensión 1280 y se les suma un embedding posicional.

3. **Teacher–Student (DINOv2)**
   - Dos ViT con la misma arquitectura:
     - **Student**: recibe vistas globales + locales (con masking). Se actualiza por gradiente.
     - **Teacher**: recibe vistas globales, se actualiza por EMA (promedio móvil de los pesos del student).
   - Las salidas del student intentan **imitar las del teacher** → pérdida de auto-supervisión.
   - Al final de esta fase nos quedamos con el **encoder entrenado**, no con un mapa de salida.

---

### 2. Decoder DPT para CHM de alta resolución (ALS)

A partir de aquí, el encoder queda **congelado** y sólo se entrena el decoder.

1. **Reassemble blocks**
   - Toman las features del ViT en distintas capas y las transforman en mapas 2D a distintas escalas.
   - Cada bloque:
     - **Read**: reordena los tokens a su posición espacial → mapa 2D.
     - **Concat + Project (Conv 1×1)**: apila canales y reduce/reorganiza la información.
     - **Resampleₛ**: ajusta el tamaño del mapa para trabajar en escalas 1/32, 1/16, 1/8 y 1/4.

2. **Fusion blocks**
   - Combinan información **global** (mapas más pequeños) con **detalle fino** (mapas de mayor resolución).
   - Cada bloque:
     - Aplica una **Residual Conv Unit** para limpiar/refinar.
     - **Suma residual** entre el mapa global y el mapa más fino.
     - Hace un **upsample ×2** (Resample₀.5) para ir subiendo de resolución.
     - Otro **Project (Conv 1×1)** adapta el número de canales para el siguiente nivel.

3. **Head (salida por bins)**
   - Toma el último mapa de features (64×64) y:
     - Aplica un **upsample** para volver a 256×256.
     - Conv 1×1 → genera **256 bins de altura por píxel**.
     - Softmax → histograma de probabilidad de altura por píxel.
     - Promedio ponderado → altura esperada en metros.
   - Se obtiene un **CHM predicho 256×256**, alineado con el tile de entrada.

4. **Función de pérdida: Sigloss**
   - Variante de la pérdida de profundidad de Eigen et al.:
     - Trabaja en espacio logarítmico.
     - Penaliza errores absolutos y errores globales de escala.
   - Se usa el **CHM ALS real** como verdad terreno.

---

### 3. Modelo GEDI global y fusión ALS + GEDI

1. **Modelo GEDI (CNN + metadata)**
   - Entrada:
     - Parche RGB de 128×128.
     - Metadatos: latitud, longitud, elevación solar, ángulo off-nadir, pendiente del terreno.
   - Arquitectura:
     - **Extractor CNN** con varias capas Conv2D + ReLU + MaxPooling.
     - **Flatten → capas densas**, donde se concatenan los metadatos.
   - Salida:
     - Un escalar: altura **RH95** (GEDI) en ese footprint.
   - Pérdida:
     - **L1 Loss** entre altura predicha y altura medida por GEDI.

2. **Cálculo de factor de reescalamiento**
   - Se cruzan las predicciones del modelo ALS y del modelo GEDI en zonas con datos comunes.
   - Se calcula un **factor de escala espacialmente suave** que corrige el CHM ALS.

3. **CHM final**
   - El CHM ALS de alta resolución se multiplica por el factor de reescalamiento.
   - Resultado: **canopy height map continuo**, detallado y coherente con GEDI a escala global.

---

## 🖥️ Interfaz de Streamlit (implementación del proyecto)

La aplicación de este repo incluye:

- **Modo NEON interno**
  - Selección de tiles del dataset NEON.
  - Visualización de:
    - Imagen aérea.
    - CHM predicho por el modelo.
    - CHM real (ALS).
  - Cálculo de métricas:
    - MAE, RMSE, R² a nivel píxel y por bloques.
    - Sesgo (Bias).
- **Modo “censo estructural”**
  - A partir del CHM predicho:
    - Altura promedio del dosel.
    - Altura P95.
    - % de área con árboles (ej. h > 1 m).
    - Distribución de alturas (histograma).
  - Todo presentado en paneles tipo dashboard.






















---

## 🧬 Tecnologías principales

- Python, PyTorch
- Vision Transformers (ViT Huge, DINOv2)
- Dense Prediction Transformer (DPT)
- Redes convolucionales (CNN)
- Streamlit para visualización interactiva

---

## 📚 Referencias

- Weinstein, B. G., et al. **High-resolution canopy height maps by learning from airborne lidar and spaceborne GEDI.**  
- Repositorio oficial: https://github.com/facebookresearch/HighResCanopyHeight
- Oquab, M., et al. **DINOv2: Learning robust visual features without supervision.**
- Ranftl, R., et al. **Vision Transformers for dense prediction (DPT).**
