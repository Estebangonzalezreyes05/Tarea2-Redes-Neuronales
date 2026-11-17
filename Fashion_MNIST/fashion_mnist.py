import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.decomposition import PCA

# ==========================================
# PROYECTO: CLASIFICACIÓN DE IMÁGENES FASHION-MNIST
# ==========================================

RUTA_RESULTADOS = "../Resultados"
os.makedirs(RUTA_RESULTADOS, exist_ok=True)

CLASES_FASHION = [
    "Camiseta/Top", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Tenis", "Bolso", "Botín"
]

print("Cargando conjunto de datos Fashion-MNIST...")

(train_ds, test_ds), info = tfds.load(
    "fashion_mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    batch_size=32,
    with_info=True
)

num_clases = info.features["label"].num_classes
print(f"Clases totales: {num_clases}")

# ==========================================
# 1. PREPARACIÓN DE DATOS (TRAIN Y TEST)
# ==========================================
print("\n[Paso 1] Extrayendo arreglos de entrenamiento y prueba...")

def preparar_datos_fashion(dataset):
    imagenes, etiquetas = [], []
    for imgs, lbls in tfds.as_numpy(dataset):
        # Fashion-MNIST ya viene orientado correctamente; solo quitamos el canal extra
        imagenes.append(imgs.squeeze(-1))
        etiquetas.append(lbls)
    return np.concatenate(imagenes, axis=0), np.concatenate(etiquetas, axis=0)

X_train, y_train = preparar_datos_fashion(train_ds)
X_test, y_test = preparar_datos_fashion(test_ds)

print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

# Muestra visual 5x5 de prendas
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(CLASES_FASHION[y_train[i]], fontsize=9)
    ax.axis("off")
plt.suptitle("Muestra de imágenes del conjunto Fashion-MNIST")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_RESULTADOS, "fashion_muestra.png"), dpi=150)
plt.close()

# ==========================================
# 2. NORMALIZACIÓN DE IMÁGENES [0, 1]
# ==========================================
print("\n[Paso 2] Normalizando píxeles...")

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print(f"Rango normalizado -> Mín: {X_train.min()}, Máx: {X_train.max()}")

# ==========================================
# 3. APLICACIÓN DE FLATTEN (28x28 -> 784)
# ==========================================
print("\n[Paso 3] Aplicando capa Flatten...")

capa_flatten = tf.keras.layers.Flatten()
X_train_flat = capa_flatten(X_train).numpy()
X_test_flat = capa_flatten(X_test).numpy()

print(f"X_train_flat: {X_train_flat.shape}")
print(f"X_test_flat:  {X_test_flat.shape}")

# ==========================================
# 4. GRÁFICA BIDIMENSIONAL CON PCA
# ==========================================
print("\n[Paso 4] Calculando proyección bidimensional con PCA...")

pca = PCA(n_components=2, random_state=42)
pca.fit(X_train_flat)

muestra = 5000
X_test_pca = pca.transform(X_test_flat[:muestra])
y_test_pca = y_test[:muestra]

varianza = pca.explained_variance_ratio_ * 100
print(f"Varianza explicada -> PC1: {varianza[0]:.2f}%, PC2: {varianza[1]:.2f}%")

plt.figure(figsize=(11, 8))
cmap_discreto = plt.get_cmap("tab10", 10)
scatter = plt.scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c=y_test_pca,
    cmap=cmap_discreto,
    vmin=-0.5,
    vmax=9.5,
    alpha=0.65,
    s=14
)
cbar = plt.colorbar(scatter, ticks=range(10))
cbar.ax.set_yticklabels(CLASES_FASHION)
plt.title("Proyección PCA 2D - Fashion-MNIST (5,000 muestras de prueba)")
plt.xlabel(f"Componente Principal 1 ({varianza[0]:.2f}% de varianza)")
plt.ylabel(f"Componente Principal 2 ({varianza[1]:.2f}% de varianza)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

ruta_pca = os.path.join(RUTA_RESULTADOS, "fashion_pca_2d.png")
plt.savefig(ruta_pca, dpi=150)
plt.close()
print("Gráfica PCA guardada exitosamente en:", ruta_pca)

# ==========================================
# 5. ENTRENAMIENTO DE 3 CONFIGURACIONES DE REDES NEURONALES
# ==========================================
print("\n[Paso 5] Construyendo y entrenando 3 configuraciones de redes neuronales...")

def crear_config_1(input_dim, num_classes):
    """Configuración 1: Red superficial (1 capa oculta de 128 neuronas)"""
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ], name="Config1_1Capa_128")

def crear_config_2(input_dim, num_classes):
    """Configuración 2: Red intermedia (2 capas ocultas: 256 y 128 neuronas)"""
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ], name="Config2_2Capas_256_128")

def crear_config_3(input_dim, num_classes):
    """Configuración 3: Red profunda con regularización Dropout (512, 256, 128)"""
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ], name="Config3_3Capas_Dropout")

modelos = [
    ("Config 1: 1 Capa (128)", crear_config_1(784, num_clases)),
    ("Config 2: 2 Capas (256-128)", crear_config_2(784, num_clases)),
    ("Config 3: 3 Capas + Dropout (512-256-128)", crear_config_3(784, num_clases))
]

EPOCHS = 15
BATCH_SIZE = 128
resultados = []

for nombre, modelo in modelos:
    print(f"\n--- Entrenando {nombre} ---")
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    historial = modelo.fit(
        X_train_flat,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    
    test_loss, test_acc = modelo.evaluate(X_test_flat, y_test, verbose=0)
    
    resultados.append({
        "nombre": nombre,
        "train_acc": historial.history["accuracy"][-1],
        "val_acc": historial.history["val_accuracy"][-1],
        "test_acc": test_acc,
        "historial": historial.history
    })

# ==========================================
# 6. COMPARACIÓN DE ACCURACY
# ==========================================
print("\n================ COMPARACIÓN DE ACCURACY (FASHION-MNIST) ================")
encabezado = f"{'Configuración':<42} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10}"
print(encabezado)
print("-" * 80)

lineas_csv = ["Configuracion,Train_Accuracy,Val_Accuracy,Test_Accuracy"]

for r in resultados:
    fila = f"{r['nombre']:<42} | {r['train_acc']*100:>8.2f}% | {r['val_acc']*100:>8.2f}% | {r['test_acc']*100:>8.2f}%"
    print(fila)
    lineas_csv.append(f"{r['nombre']},{r['train_acc']*100:.2f}%,{r['val_acc']*100:.2f}%,{r['test_acc']*100:.2f}%")

ruta_csv = os.path.join(RUTA_RESULTADOS, "fashion_resultados.csv")
with open(ruta_csv, "w", encoding="utf-8") as f:
    f.write("\n".join(lineas_csv))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for r in resultados:
    ax1.plot(range(1, EPOCHS + 1), r["historial"]["val_accuracy"], marker="o", label=r["nombre"])
ax1.set_title("Fashion-MNIST: Evolución de Validation Accuracy")
ax1.set_xlabel("Época")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.4)

nombres = [r["nombre"] for r in resultados]
test_accs = [r["test_acc"] * 100 for r in resultados]
colores = ["#4C72B0", "#55A868", "#C44E52"]

barras = ax2.bar(nombres, test_accs, color=colores, width=0.5)
ax2.set_title("Fashion-MNIST: Comparación de Accuracy en Prueba (Test)")
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(0, 100)
ax2.tick_params(axis="x", rotation=10)

for barra in barras:
    altura = barra.get_height()
    ax2.text(
        barra.get_x() + barra.get_width() / 2,
        altura + 1,
        f"{altura:.2f}%",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

plt.tight_layout()
ruta_comp = os.path.join(RUTA_RESULTADOS, "fashion_comparacion_accuracy.png")
plt.savefig(ruta_comp, dpi=150)
plt.close()

print(f"\nTabla guardada en: {ruta_csv}")
print(f"Gráfica comparativa guardada en: {ruta_comp}")