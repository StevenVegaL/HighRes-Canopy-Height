
<p align="center">
  <img src="app/assets/banner_chm.png" width="100%" />
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


## 🏗️ Resumen teórico de la arquitectura

Aquí se resume la arquitectura completa en 3 niveles: **encoder SSL**, **decoder ALS** y **modelo GEDI + fusión**.

---

### 1. Encoder SSL: ViT Huge con DINOv2

<p align="center">
  <img src="app/assets/vit.png" width="60%" />
</p>


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

<p align="center">
  <img src="app/assets/vit2.png" width="100%" />
</p>

     - **Student**: recibe vistas globales + locales (con masking). Se actualiza por gradiente.
     - **Teacher**: recibe vistas globales, se actualiza por EMA (promedio móvil de los pesos del student).
   - Las salidas del student intentan **imitar las del teacher** → pérdida de auto-supervisión.
   - Al final de esta fase nos quedamos con el **encoder entrenado**, no con un mapa de salida.

---

### 2. Decoder DPT para CHM de alta resolución (ALS)

<p align="center">
  <img src="app/assets/dpt.png" width="100%" />
</p>

A partir de aquí, el encoder queda **congelado** y sólo se entrena el decoder.

1. **Reassemble blocks**

<p align="center">
  <img src="app/assets/rem.png" width="60%" />
</p>

   - Toman las features del ViT en distintas capas y las transforman en mapas 2D a distintas escalas.
   - Cada bloque:
     - **Read**: reordena los tokens a su posición espacial → mapa 2D.
     - **Concat + Project (Conv 1×1)**: apila canales y reduce/reorganiza la información.
     - **Resampleₛ**: ajusta el tamaño del mapa para trabajar en escalas 1/32, 1/16, 1/8 y 1/4.

2. **Fusion blocks**

<p align="center">
  <img src="app/assets/fus.png" width="60%" />
</p>

   - Combinan información **global** (mapas más pequeños) con **detalle fino** (mapas de mayor resolución).
   - Cada bloque:
     - Aplica una **Residual Conv Unit** para limpiar/refinar.
     - **Suma residual** entre el mapa global y el mapa más fino.
     - Hace un **upsample ×2** (Resample₀.5) para ir subiendo de resolución.
     - Otro **Project (Conv 1×1)** adapta el número de canales para el siguiente nivel.

3. **Head (salida por bins)**

<p align="center">
  <img src="app/assets/head.png" width="60%" />
</p>

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

<p align="center">
  <img src="app/assets/gedi.png" width="100%" />
</p>

1. **Modelo GEDI (CNN + metadata)**
<p align="center">
  <img src="app/assets/cnn.png" width="100%" />
</p>

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

<p align="center">
  <img src="app/assets/combi.png" width="100%" />
</p>
   - Se cruzan las predicciones del modelo ALS y del modelo GEDI en zonas con datos comunes.
   - Se calcula un **factor de escala espacialmente suave** que corrige el CHM ALS.

3. **CHM final**
   - El CHM ALS de alta resolución se multiplica por el factor de reescalamiento.
   - Resultado: **canopy height map continuo**, detallado y coherente con GEDI a escala global.

---






### ⚖️🌳 Descarga de pesos preentrenados 

Para que la aplicación pueda realizar **inferencia real**, es indispensable descargar los **pesos preentrenados** del modelo original de Meta AI:  
**High-Resolution Canopy Height Maps**.



#### ¿De dónde descargar los pesos?

1. Ve al repositorio original del proyecto (Meta / `HighResolutionCanopyHeight`).
2. Busca la sección de **model checkpoints / weights**.
3. Descarga, como mínimo, los siguientes archivos:

- ✅ **Checkpoint del modelo CHM**, por ejemplo:  
  `compressed_SSLhuge_aerial.pth`

- ✅ **Pesos de la red de normalización RNet**, usados cuando `normtype = 2`.  
  El nombre del archivo debe coincidir con lo que espera la función  
  `load_rnet_normalizer()` en `model/ssl_model.py`.


#### 3.2. Dónde ubicar los archivos descargados

Copia los archivos descargados en la carpeta:

```bash
saved_checkpoints/
```

---

### 🚀 Ejecución del Proyecto con **Docker**
Instalación • Despliegue • Uso

Este proyecto está preparado para ejecutarse fácilmente usando **Docker**, sin necesidad de instalar manualmente todas las dependencias en tu máquina local.



#### 📁 Clonar el repositorio

```bash
git clone https://github.com/StevenVegaL/HighRes-Canopy-Height
cd HighResCanopyHeightApp
```

⚠️ Importante:
Antes de continuar, asegúrate de que la carpeta saved_checkpoints/ contiene los pesos indicados en la sección anterior (modelo CHM y RNet).


#### 🛠️ Construir la imagen Docker
Desde la raíz del proyecto, ejecuta:

```bash

docker build -t chm-demo .

```



🔎 ¿Qué hace este comando?

Elemento	Descripción
-t chm-demo	Asigna el nombre chm-demo a la imagen Docker
.	Usa el Dockerfile ubicado en el directorio actual







####  ▶️ Ejecutar el contenedor
Una vez construida la imagen, puedes levantar el contenedor con:

```bash
docker run -p 8501:8501 chm-demo
```


💡 Si el puerto 8501 ya está ocupado en tu máquina, puedes usar otro puerto externo, por ejemplo:

```bash
docker run -p 8502:8501 chm-demo
```




#### 🌐 Acceder a la aplicación
Con el contenedor en ejecución, abre tu navegador en:

```bash
http://localhost:8501
```
Deberías ver la landing de la aplicación.




---

### 💻 Ejecución local (opcional, sin Docker)

Aunque la forma recomendada de ejecutar el proyecto es mediante **Docker**, también puedes correr la aplicación **localmente** si ya tienes **Python** instalado en tu máquina.



#### 🧬 Crear entorno virtual e instalar dependencias

Se recomienda usar un entorno virtual para aislar las dependencias del proyecto.

#### 1️⃣ Crear y activar el entorno virtual

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En Linux / macOS:

```bash

source .venv/bin/activate
```

Verás que el prompt de tu terminal cambia, indicando que el entorno .venv está activo.

2️⃣ Actualizar pip e instalar dependencias
Con el entorno virtual activado, ejecuta:

```bash
pip install --upgrade pip
pip install -r requirements.txt

```



#### 🚀  Lanzar la aplicación con Streamlit
Una vez instaladas las dependencias, desde la raíz del proyecto ejecuta:

```bash

streamlit run app/streamlit_landing_CHM_app.py

```

Si todo está correctamente configurado (incluyendo los pesos en saved_checkpoints/), Streamlit levantará la aplicación.

🌐 Acceder a la app
Abre tu navegador y visita:

```bash
http://localhost:8501
```

Allí ceberías ver la landing de la aplicación.

---

### 🧠 Explicación: ¿cómo se cargan los pesos y cómo se realiza la inferencia?

La lógica de carga de pesos y de inferencia está dividida en dos contextos:


#### 🌲 Modo NEON (dataset) – usa RNet + NeonDataset.


#### 🖼️ Modo de imagen subida – usa solo el modelo CHM con normalización global.






---

## 📚 Referencias

- Weinstein, B. G., et al. **High-resolution canopy height maps by learning from airborne lidar and spaceborne GEDI.**  
- Repositorio oficial: https://github.com/facebookresearch/HighResCanopyHeight
- Oquab, M., et al. **DINOv2: Learning robust visual features without supervision.**
- Ranftl, R., et al. **Vision Transformers for dense prediction (DPT).**
