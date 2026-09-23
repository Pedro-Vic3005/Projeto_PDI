"""Canny com mediana e Otsu; imprime as cinco métricas do artigo.

Execute: python canny_simplificado_metricas.py [imagem.png] [bordas_referencia.png]
"""

from pathlib import Path
import csv
import sys

import cv2
import numpy as np


def calcular_metricas(original, restaurada, bordas, referencia):
    """MSE/SNR em tons de cinza; TPR/TNR/ACC nos mapas de bordas."""
    if not (original.shape == restaurada.shape == bordas.shape == referencia.shape):
        raise ValueError("Todas as imagens devem ter a mesma largura e altura.")

    limpa = original.astype(np.float64)
    recuperada = restaurada.astype(np.float64)
    mse = float(np.mean((recuperada - limpa) ** 2))
    variancia = float(np.var(recuperada))
    if mse == 0:
        snr = float("inf") if variancia > 0 else float("nan")
    elif variancia == 0:
        snr = float("-inf")
    else:
        snr = float(10 * np.log10(variancia / mse))

    prevista = bordas > 0
    real = referencia > 0
    tp = int(np.count_nonzero(prevista & real))
    tn = int(np.count_nonzero(~prevista & ~real))
    fp = int(np.count_nonzero(prevista & ~real))
    fn = int(np.count_nonzero(~prevista & real))

    def dividir(numerador, denominador):
        return numerador / denominador if denominador else float("nan")

    return {
        "MSE": mse,
        "SNR_dB": snr,
        "TPR": dividir(tp, tp + fn),
        "TNR": dividir(tn, tn + fp),
        "ACC": dividir(tp + tn, tp + tn + fp + fn),
    }


# 1. Carregar a imagem em tons de cinza.
imagem = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\aluno\Desktop\pvga\imagens\lena.png")
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
saida = Path(__file__).with_name("resultados_simplificado")
saida.mkdir(exist_ok=True)
cv2.imwrite(str(saida / "1_original.png"), original)
cv2.imwrite(str(saida / "2_ruido.png"), ruidosa)
cv2.imwrite(str(saida / "3_tradicional.png"), tradicional)
cv2.imwrite(str(saida / "4_melhorado.png"), melhorado)
painel = np.vstack((np.hstack((original, ruidosa)),
                    np.hstack((tradicional, melhorado))))
cv2.imwrite(str(saida / "comparacao.png"), painel)

# Referência anotada opcional; sem ela, Canny na imagem limpa é só um proxy.
# Esse proxy favorece métodos parecidos com o Canny tradicional.
if len(sys.argv) > 2:
    referencia = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    if referencia is None:
        raise FileNotFoundError(f"Não foi possível abrir o mapa de bordas {sys.argv[2]}")
    tipo_referencia = "mapa de bordas fornecido"
else:
    referencia = cv2.Canny(cv2.GaussianBlur(original, (5, 5), 1), 50, 100)
    tipo_referencia = "Canny da imagem limpa (referência aproximada)"

cv2.imwrite(str(saida / "6_referencia_bordas.png"), referencia)
metricas = {
    "tradicional": calcular_metricas(original, suave, tradicional, referencia),
    "modificado": calcular_metricas(original, filtrada, melhorado, referencia),
}
with (saida / "metricas.csv").open("w", newline="", encoding="utf-8") as arquivo:
    campos = ["metodo", *next(iter(metricas.values())).keys()]
    escritor = csv.DictWriter(arquivo, fieldnames=campos)
    escritor.writeheader()
    for metodo, valores in metricas.items():
        escritor.writerow({"metodo": metodo, **valores})

print(f"Limiar alto por Otsu: {alto:.1f}; baixo: {alto / 2:.1f}")
print(f"Referência de bordas: {tipo_referencia}")
for metodo, valores in metricas.items():
    print(f"{metodo}: " + ", ".join(f"{nome}={valor:.4f}" for nome, valor in valores.items()))
print(f"Abra {saida / 'comparacao.png'}")
