# Dataset MNIST

El programa espera los 4 archivos del dataset MNIST en formato IDX dentro de este
directorio (`data/`):

```
data/
├── train-images-idx3-ubyte   (60000 imágenes de entrenamiento)
├── train-labels-idx1-ubyte   (60000 etiquetas de entrenamiento)
├── t10k-images-idx3-ubyte    (10000 imágenes de prueba)
└── t10k-labels-idx1-ubyte    (10000 etiquetas de prueba)
```

## Descarga

Los archivos pueden descargarse desde el sitio original o desde un mirror:

- Sitio original: http://yann.lecun.com/exdb/mnist/
- Mirror en Kaggle: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Descarga los 4 archivos `.gz`, descomprímelos y colócalos en este directorio:

```bash
cd data
# Ejemplo con los archivos ya descargados:
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
```

## Formato IDX (big-endian)

- **Imágenes**: `magic(4) + count(4) + rows(4) + cols(4)` + `count * rows * cols` bytes.
- **Etiquetas**: `magic(4) + count(4)` + `count` bytes.

Cada píxel es un byte (0–255) que el lector normaliza a `[0,1]` dividiéndolo entre 255.
