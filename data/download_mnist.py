#!/usr/bin/env python3
"""
Script helper para descargar el dataset MNIST en formato IDX binario.
"""
import os
import sys
import urllib.request
import gzip
import shutil

URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    os.makedirs(target_dir, exist_ok=True)

    print(f"Descargando dataset MNIST a: {os.path.abspath(target_dir)}")

    for gz_name, url in URLS.items():
        decompressed_name = gz_name.replace(".gz", "")
        dest_path = os.path.join(target_dir, decompressed_name)

        if os.path.exists(dest_path):
            print(f"  [OK] Ya existe: {decompressed_name}")
            continue

        gz_path = os.path.join(target_dir, gz_name)
        print(f"  [>] Descargando {gz_name}...")
        urllib.request.urlretrieve(url, gz_path)

        print(f"  [>] Descomprimiendo {decompressed_name}...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(dest_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(gz_path)
        print(f"  [OK] Completado: {decompressed_name}")

    print("\nDataset MNIST listo para su uso.")

if __name__ == "__main__":
    main()
