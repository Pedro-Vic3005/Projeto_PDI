"""Primeiro experimento: Canny comum e Canny com mediana + Otsu."""

from pathlib import Path

import cv2
import numpy as np


# 1. Carregar a imagem em tons de cinza.
imagem = Path("C:/Users/aluno/Desktop/pvga/imagens/lena.png")
original = cv2.imread(str(imagem), cv2.IMREAD_GRAYSCALE)
if original is None:
    raise FileNotFoundError(f"Não foi possível abrir {imagem}")

# 2. Acrescentar ruído sal e pimenta em 2% dos pixels.
ruidosa = original.copy()
gerador = np.random.default_rng(42)
posicoes = gerador.choice(ruidosa.size, round(ruidosa.size * 0.02), replace=False)
pixels = ruidosa.reshape(-1)
meio = len(posicoes) // 2
pixels[posicoes[:meio]] = 0       # pimenta: preto
pixels[posicoes[meio:]] = 255     # sal: branco

# 3. Canny tradicional: filtro gaussiano e limiares escolhidos manualmente.
suave = cv2.GaussianBlur(ruidosa, (5, 5), 1)
tradicional = cv2.Canny(suave, 50, 100)

# 4. Primeira melhoria: trocar o gaussiano por mediana 3x3.
filtrada = cv2.medianBlur(ruidosa, 3)

# 5. Segunda melhoria: Otsu calcula o limiar alto a partir do gradiente.
gx = cv2.Sobel(filtrada, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(filtrada, cv2.CV_32F, 0, 1, ksize=3)
forca = cv2.magnitude(gx, gy)
maximo = float(forca.max())
if maximo > 0:
    forca_8bits = np.uint8(forca * 255 / maximo)
    otsu, _ = cv2.threshold(forca_8bits, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    alto = max(1.0, float(otsu) * maximo / 255)
    melhorado = cv2.Canny(filtrada, alto / 2, alto)
else:
    alto = 0.0
    melhorado = np.zeros_like(original)

# 6. Salvar as imagens. O painel fica na ordem:
# original | com ruído / Canny tradicional | Canny com mediana + Otsu.
saida = Path(__file__).with_name("resultados")
saida.mkdir(exist_ok=True)
cv2.imwrite(str(saida / "1_original.png"), original)
cv2.imwrite(str(saida / "2_ruido.png"), ruidosa)
cv2.imwrite(str(saida / "3_filtrada.png"), filtrada)
cv2.imwrite(str(saida / "4_tradicional.png"), tradicional)
cv2.imwrite(str(saida / "5_melhorado.png"), melhorado)
painel = np.vstack((np.hstack((original, ruidosa)),
                    np.hstack((tradicional, melhorado))))
cv2.imwrite(str(saida / "comparacao.png"), painel)
print(f"Limiar alto por Otsu: {alto:.1f}; baixo: {alto / 2:.1f}")
print(f"Abra {saida / 'comparacao.png'}")
