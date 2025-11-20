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



---


### 3. Descarga de pesos preentrenados ⚖️🌳

Para que la aplicación pueda realizar **inferencia real**, es indispensable descargar los **pesos preentrenados** del modelo original de Meta AI:  
**High-Resolution Canopy Height Maps**.



#### 3.1. ¿De dónde descargar los pesos?

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

## 🚀 4. Ejecución del Proyecto con **Docker**
Instalación • Despliegue • Uso

Este proyecto está preparado para ejecutarse fácilmente usando **Docker**, sin necesidad de instalar manualmente todas las dependencias en tu máquina local.



### 📁 4.1. Clonar el repositorio

```bash
git clone <URL_DE_TU_REPOSITORIO>
cd HighResCanopyHeightApp
```

⚠️ Importante:
Antes de continuar, asegúrate de que la carpeta saved_checkpoints/ contiene los pesos indicados en la sección anterior (modelo CHM y RNet).


🛠️ 4.2. Construir la imagen Docker
Desde la raíz del proyecto, ejecuta:

```bash

docker build -t chm-demo .

```



🔎 ¿Qué hace este comando?

Elemento	Descripción
-t chm-demo	Asigna el nombre chm-demo a la imagen Docker
.	Usa el Dockerfile ubicado en el directorio actual



📦 El Dockerfile se encarga de:

Instalar Python y las dependencias necesarias (PyTorch, PyTorch Lightning, Streamlit, etc.).

Copiar el código fuente (app/, model/, utils/, etc.) dentro del contenedor.

Asegurar el acceso a saved_checkpoints/ para cargar los pesos del modelo.

Definir el comando de arranque de Streamlit como punto de entrada.




▶️ 4.3. Ejecutar el contenedor
Una vez construida la imagen, puedes levantar el contenedor con:

```bash
docker run -p 8501:8501 chm-demo
```


💡 Si el puerto 8501 ya está ocupado en tu máquina, puedes usar otro puerto externo, por ejemplo:

```bash
docker run -p 8502:8501 chm-demo
```




🌐 4.4. Acceder a la aplicación
Con el contenedor en ejecución, abre tu navegador en:

```bash
http://localhost:8501
```
Deberías ver la landing de la aplicación.

Desde allí puedes:

Navegar al modo “Demostración” usando el menú superior.

Explorar tiles reales del dataset NEON.

Visualizar la imagen aérea, el CHM real y el CHM predicho por el modelo.





## 💻 5. Ejecución local (opcional, sin Docker)

Aunque la forma recomendada de ejecutar el proyecto es mediante **Docker**, también puedes correr la aplicación **localmente** si ya tienes **Python** instalado en tu máquina.

---

### 🧬 5.1. Crear entorno virtual e instalar dependencias

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

Esto instalará todas las librerías necesarias para:

Cargar el modelo CHM y la red de normalización RNet.

Ejecutar la interfaz de Streamlit.

Trabajar con imágenes, tensores y métricas del modelo.

🚀 5.2. Lanzar la aplicación con Streamlit
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

Allí podrás:

Ver la landing del proyecto.

Acceder al modo Demostración.

Explorar los tiles del dataset NEON o las opciones que hayas habilitado en la app.


6. Explicación: ¿cómo se cargan los pesos y cómo se realiza la inferencia?

La lógica de carga de pesos y de inferencia está dividida en dos contextos:

Modo NEON (dataset) – usa RNet + NeonDataset.

Modo de imagen subida – usa solo el modelo CHM con normalización global.

6.1. Modo NEON (dataset)

La lógica principal está en model/inference_neon_tile.py y en la página app/pages/Demostración.py.

6.1.1. Configuración de componentes (setup_neon_inference)

En inference_neon_tile.py:

components = setup_neon_inference(
    checkpoint_name="compressed_SSLhuge_aerial.pth",
    normtype=2,
    trained_rgb=False,
    src_img="neon",
)


Esta función:

Carga la red de normalización RNet (si normtype == 2) mediante:

model_norm = load_rnet_normalizer()


Construye el NeonDataset:

dataset = build_neon_dataset(
    model_norm=model_norm,
    normtype=normtype,
    trained_rgb=trained_rgb,
    src_img=src_img,
)


Aquí se aplica la normalización de dominio descrita en el paper para que las imágenes NEON queden en un espacio similar al de entrenamiento del backbone.

Carga el modelo de altura de dosel (CHM):

model, device = load_chm_model(checkpoint_name=checkpoint_name)


Esto activa el modelo DINOv2 + DPT que predice alturas en metros.

Define la normalización global por canal:

norm = T.Normalize(
    mean=(0.420, 0.411, 0.296),
    std=(0.213, 0.156, 0.143),
)


Es la misma normalización utilizada en el script de inferencia original.

El resultado es un diccionario:

components = {
    "model": model,
    "device": device,
    "dataset": dataset,
    "norm": norm,
}


que la app reutiliza para todos los tiles.

6.1.2. Inferencia sobre un tile (run_neon_tile_inference)

Cuando el usuario selecciona un índice y pulsa “⚡ Calcular CHM para este tile”, en Demostración.py se llama:

result = run_neon_tile_inference(components, idx)


Dentro de run_neon_tile_inference:

Obtiene el sample del dataset:

img_no_norm, img_norm, chm = get_neon_sample(dataset, index)


img_no_norm: imagen RGB original.

img_norm: imagen ya ajustada por RNet / normalización de dominio.

chm: CHM real (LiDAR).

Prepara el batch e incluye la normalización global:

x = img_norm.unsqueeze(0)  # [1, 3, H, W]
x = norm(x)
x = x.to(device)


Ejecuta el modelo CHM:

model.eval()
with torch.no_grad():
    pred = model(x)          # [1, 1, H, W]
    pred = pred.cpu().relu()
    pred_map = pred[0, 0].numpy()  # [H, W]


Recupera el CHM real:

chm_map = chm[0].numpy()


Calcula las métricas:

metrics = compute_all_metrics(pred_map, chm_map)


Que incluye: MAE, RMSE, R² pixel, R² por bloques, Bias, etc.

Prepara la imagen RGB para mostrarla:

img_rgb = np.moveaxis(img_no_norm.numpy(), 0, 2)  # [H, W, 3]


Devuelve:

result = {
    "img_rgb": img_rgb,
    "chm_gt": chm_map,
    "chm_pred": pred_map,
    "metrics": metrics,
}


En la app, chm_pred y chm_gt se normalizan a [0,1] y se convierten a mapas de color con un colormap tipo viridis para mostrarlos como imágenes.

### 🖼️ 6.2. Modo de imagen subida

En este modo **no se usa RNet**: se asume que las imágenes subidas por el usuario son razonablemente similares al dominio NEON (imágenes aéreas, alta resolución, etc.).

La lógica principal está en el bloque `else:` de:

- `app/pages/Demostración.py`

---

#### ⚙️ 6.2.1. Carga del modelo

Para este modo se prepara un conjunto de componentes más simple:

```python
model, device = load_chm_model(checkpoint_name="compressed_SSLhuge_aerial.pth")

norm = T.Normalize(
    mean=(0.420, 0.411, 0.296),
    std=(0.213, 0.156, 0.143),
)
En resumen:

Se carga el modelo CHM (backbone DINOv2 + decoder DPT).

Se define la normalización global por canal, igual a la usada en el script de inferencia original.

No se construye NeonDataset ni se aplica RNet.

🔁 6.2.2. Flujo de inferencia
El usuario puede subir:

rgb_file: imagen aérea RGB.

chm_file (opcional): raster de CHM real, co-registrado con la imagen RGB.

1️⃣ Procesamiento de la imagen RGB
La imagen se transforma a tensor normalizado antes de entrar al modelo:

python
Copiar código
rgb_img = Image.open(rgb_file).convert("RGB")
img_np = np.array(rgb_img).astype("float32") / 255.0  # [H, W, 3]

img_t = torch.from_numpy(img_np).permute(2, 0, 1)     # [3, H, W]
x = img_t.unsqueeze(0)                                # [1, 3, H, W]
x = norm(x).to(device)
Pasos clave:

Se abre la imagen y se asegura el modo RGB.

Se normaliza a rango [0, 1].

Se permutan las dimensiones a formato [C, H, W].

Se añade la dimensión de batch → [1, 3, H, W].

Se aplica la normalización global norm y se envía a la device.

2️⃣ Predicción del CHM
Se ejecuta el modelo para obtener el mapa de altura predicho:

python
Copiar código
with torch.no_grad():
    pred = model(x)
    pred = pred.cpu().relu()[0, 0].numpy()  # [H, W]

chm_pred_up = pred
Se desactiva el gradiente (torch.no_grad()).

El modelo devuelve un tensor [1, 1, H, W].

Se lleva a CPU, se aplica relu() (sin alturas negativas) y se extrae el mapa [H, W].

3️⃣ (Opcional) Uso de un CHM real para evaluación
Si el usuario también sube un archivo de CHM real:

python
Copiar código
chm_img = Image.open(chm_file)
chm_arr = np.array(chm_img).astype("float32")

if chm_arr.ndim == 3:
    chm_arr = chm_arr[..., 0]

if chm_arr.shape != chm_pred_up.shape:
    raise ValueError(
        f"Dimensiones distintas entre predicción {chm_pred_up.shape} "
        f"y CHM real {chm_arr.shape}. Deben coincidir."
    )
Se carga el raster de CHM.

Si viene con 3 canales, se toma solo uno.

Se valida que el tamaño del CHM real coincida con el de la predicción; si no, se lanza un error.

Solo cuando las dimensiones coinciden se calculan las métricas:

python
Copiar código
metrics = compute_all_metrics(chm_pred_up, chm_arr)
👀 6.2.3. Qué muestra la app en este modo
La interfaz visualiza:

✅ Imagen RGB subida por el usuario.

✅ CHM predicho por el modelo (convertido a mapa de color).

✅ CHM real, si fue proporcionado y tiene el mismo tamaño.

✅ Una tabla de métricas (MAE, RMSE, R², Bias, etc.) cuando se proporciona un CHM real válido.

De esta manera, el usuario puede:

Probar el modelo con sus propias imágenes.

Comparar la predicción del modelo contra un CHM real (si lo tiene).

Evaluar cuantitativamente el desempeño mediante las métricas mostradas en la app



---

## 📚 Referencias

- Weinstein, B. G., et al. **High-resolution canopy height maps by learning from airborne lidar and spaceborne GEDI.**  
- Repositorio oficial: https://github.com/facebookresearch/HighResCanopyHeight
- Oquab, M., et al. **DINOv2: Learning robust visual features without supervision.**
- Ranftl, R., et al. **Vision Transformers for dense prediction (DPT).**
