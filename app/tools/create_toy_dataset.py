#!/usr/bin/env python3
"""
create_toy_dataset.py

Gera um dataset mínimo em formato YOLO (detecção) com 1 classe ("box"),
cria imagens sintéticas com OpenCV e labels em formato YOLO, e compacta
em um ZIP pronto para upload no painel.

Estrutura final dentro do ZIP (sem pasta raiz):
- data.yaml
- images/train/*.jpg
- images/val/*.jpg
- labels/train/*.txt
- labels/val/*.txt

Uso:
  python3 app/tools/create_toy_dataset.py

Saída:
  /Users/edbentto/Eletrons_Vision_V1/app/data/zips/toy_yolo_dataset.zip

Requisitos:
- opencv-python (já presente em requirements.txt)

"""
from __future__ import annotations

import os
import random
import zipfile
from pathlib import Path
from typing import Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

# Configurações
WORKSPACE = Path("/Users/edbentto/Eletrons_Vision_V1")
BASE_TMP = WORKSPACE / "app" / "data" / "toy_dataset_tmp"
ZIP_OUT = WORKSPACE / "app" / "data" / "zips" / "toy_yolo_dataset.zip"

IMAGE_SIZE: Tuple[int, int] = (640, 640)  # (width, height)
TRAIN_IMAGES = 16
VAL_IMAGES = 4
CLASS_NAME = "box"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def ensure_dirs() -> None:
    """Cria diretórios temporários para imagens e labels (train/val)."""
    (BASE_TMP / "images" / "train").mkdir(parents=True, exist_ok=True)
    (BASE_TMP / "images" / "val").mkdir(parents=True, exist_ok=True)
    (BASE_TMP / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (BASE_TMP / "labels" / "val").mkdir(parents=True, exist_ok=True)


def gen_image_and_label(dst_img: Path, dst_lbl: Path, img_size: Tuple[int, int]) -> None:
    """Gera uma imagem com um retângulo e sua label YOLO correspondente.

    Args:
        dst_img: caminho de saída da imagem (.jpg)
        dst_lbl: caminho de saída da label (.txt)
        img_size: (width, height)
    """
    w, h = img_size
    # Fundo: cinza claro
    img = np.full((h, w, 3), 220, dtype=np.uint8)

    # Retângulo aleatório
    rect_w = random.randint(int(0.15 * w), int(0.45 * w))
    rect_h = random.randint(int(0.15 * h), int(0.45 * h))
    x1 = random.randint(0, w - rect_w)
    y1 = random.randint(0, h - rect_h)
    x2 = x1 + rect_w
    y2 = y1 + rect_h

    # Desenhar retângulo
    color = (0, 140, 255)  # laranja
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)

    # Alguns elementos decorativos (linhas) para variabilidade
    for _ in range(random.randint(2, 5)):
        p1 = (random.randint(0, w - 1), random.randint(0, h - 1))
        p2 = (random.randint(0, w - 1), random.randint(0, h - 1))
        cv2.line(img, p1, p2, (60, 60, 60), thickness=random.randint(1, 3))

    # Calcular label YOLO normalizada: class x_center y_center width height
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    label_line = f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n"

    # Salvar
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    dst_lbl.write_text(label_line, encoding="utf-8")


def write_data_yaml() -> None:
    """Gera data.yaml com 1 classe e caminhos relativos para train/val."""
    yaml_text = (
        "nc: 1\n"
        f"names: [{CLASS_NAME}]\n"
        "train: images/train\n"
        "val: images/val\n"
    )
    (BASE_TMP / "data.yaml").write_text(yaml_text, encoding="utf-8")


def build_dataset() -> None:
    ensure_dirs()

    # Train
    for i in range(TRAIN_IMAGES):
        img_p = BASE_TMP / "images" / "train" / f"img_{i:03d}.jpg"
        lbl_p = BASE_TMP / "labels" / "train" / f"img_{i:03d}.txt"
        gen_image_and_label(img_p, lbl_p, IMAGE_SIZE)

    # Val
    for i in range(VAL_IMAGES):
        img_p = BASE_TMP / "images" / "val" / f"img_{i:03d}.jpg"
        lbl_p = BASE_TMP / "labels" / "val" / f"img_{i:03d}.txt"
        gen_image_and_label(img_p, lbl_p, IMAGE_SIZE)

    write_data_yaml()


def zip_dataset() -> None:
    """Compacta o conteúdo de BASE_TMP em ZIP sem pasta raiz (arquivos na raiz do ZIP)."""
    ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in BASE_TMP.rglob("*"):
            if path.is_file():
                # arcname relativo à BASE_TMP para manter raiz limpa
                arcname = path.relative_to(BASE_TMP)
                zf.write(path, arcname)


def main() -> None:
    # Limpeza opcional: recriar pasta temporária
    if BASE_TMP.exists():
        # Remover conteúdos antigos
        for p in sorted(BASE_TMP.rglob("*"), reverse=True):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass
            else:
                try:
                    p.rmdir()
                except Exception:
                    pass
    BASE_TMP.mkdir(parents=True, exist_ok=True)

    build_dataset()
    zip_dataset()

    print("Toy dataset gerado com sucesso:")
    print(f"- Pasta temporária: {BASE_TMP}")
    print(f"- ZIP para upload: {ZIP_OUT}")
    print("\nNo painel, selecione este ZIP, use device=cpu e epochs=5-10 para validar.")


if __name__ == "__main__":
    main()