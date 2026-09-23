"""Experimento inspirado em Li e Liu (ITAIC 2022): Canny modificado.

Instale: python -m pip install opencv-python numpy
Execute: python canny_simples.py [caminho/para/imagem.png]

As etapas ambíguas do artigo estão identificadas nos comentários. Os números
da Tabela I não são reproduzíveis sem as imagens e os parâmetros originais.
"""

from pathlib import Path
import sys

import cv2
import numpy as np


# Parâmetros não informados no artigo: valores escolhidos para este experimento.
LIMIAR_FILTRO = 35
DENSIDADE_RUIDO = 0.02
SEMENTE = 42


def filtro_seletivo(imagem, T):
    """Seção III-A: examina somente centros com valor 0 ou 255.

    Interpretação das etapas 3-5: se todos os vizinhos diferem muito, substitui
    pela mediana. No caso misto, verifica se um vizinho parecido pertence a
    uma região com mais de 3 vizinhos parecidos; se sim, preserva o centro.
    O artigo não fixa T e suas condições de máximo/mínimo são ambíguas.
    """
    saida = imagem.copy()
    borda = cv2.copyMakeBorder(imagem, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    candidatos = np.argwhere((imagem == 0) | (imagem == 255))

    for y, x in candidatos:
        janela = borda[y:y + 3, x:x + 3]
        centro = int(imagem[y, x])
        diferencas = np.abs(janela.astype(np.int16) - centro)
        diferencas[1, 1] = 256  # Ignora o próprio centro.

        if np.all(diferencas[diferencas != 256] < T):
            continue

        proximos = np.argwhere(diferencas < T)
        preservar_borda = False
        if len(proximos) > 0:
            # Escolhe o vizinho cujo valor é mais próximo do pixel central.
            py, px = min(proximos, key=lambda p: diferencas[p[0], p[1]])
            vizinho = int(janela[py, px])
            # Janela 3x3 centrada nesse vizinho na imagem original.
            ao_redor = borda[y + py - 1:y + py + 2, x + px - 1:x + px + 2]
            if ao_redor.shape == (3, 3):
                qtd_parecidos = np.count_nonzero(
                    np.abs(ao_redor.astype(np.int16) - vizinho) < T
                ) - 1  # Exclui o próprio vizinho.
                preservar_borda = qtd_parecidos > 3

        if not preservar_borda:
            saida[y, x] = np.median(janela)

    return saida


def suprimir_nao_maximos(magnitude, dx, dy):
    """Mantém máximos ao longo da direção do gradiente, por interpolação."""
    comprimento = np.hypot(dx, dy)
    ux = np.divide(dx, comprimento, out=np.zeros_like(dx), where=comprimento > 0)
    uy = np.divide(dy, comprimento, out=np.zeros_like(dy), where=comprimento > 0)
    y, x = np.indices(magnitude.shape, dtype=np.float32)
    anterior = cv2.remap(magnitude, x - ux, y - uy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    seguinte = cv2.remap(magnitude, x + ux, y + uy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    afinado = np.where((magnitude >= anterior) & (magnitude >= seguinte),
                       magnitude, 0).astype(np.float32)
    afinado[[0, -1], :] = 0
    afinado[:, [0, -1]] = 0
    return afinado


def ligar_bordas(afinado, baixo, alto):
    """Histerese: mantém bordas fracas ligadas às fortes por 8 vizinhos."""
    fortes = (afinado >= alto) & (afinado > 0)
    candidatos = ((afinado >= baixo) & (afinado > 0)).astype(np.uint8)
    _, regioes = cv2.connectedComponents(candidatos, connectivity=8)
    rotulos_fortes = np.unique(regioes[fortes])
    rotulos_fortes = rotulos_fortes[rotulos_fortes != 0]
    return np.isin(regioes, rotulos_fortes).astype(np.uint8) * 255


# 1. Carregar imagem e acrescentar ruído sal e pimenta (densidade 0,02).
imagem = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\aluno\Desktop\pvga\imagens\lena.png")
original = cv2.imread(str(imagem), cv2.IMREAD_GRAYSCALE)
if original is None:
    raise FileNotFoundError(f"Não foi possível abrir {imagem}. Passe o caminho da imagem ao executar.")
ruidosa = original.copy()
gerador = np.random.default_rng(SEMENTE)
posicoes = gerador.choice(ruidosa.size, round(ruidosa.size * DENSIDADE_RUIDO), replace=False)
pixels = ruidosa.reshape(-1)
metade = len(posicoes) // 2
pixels[posicoes[:metade]] = 0
pixels[posicoes[metade:]] = 255

# 2. Comparação: Canny convencional. Estes parâmetros não constam do artigo.
gaussiana = cv2.GaussianBlur(ruidosa, (5, 5), 1)
tradicional = cv2.Canny(gaussiana, 50, 100)

# 3. Filtro seletivo 3x3 em lugar do gaussiano.
filtrada = filtro_seletivo(ruidosa, LIMIAR_FILTRO)

# 4. Quatro máscaras da equação (8): x, y, 45° e 135°.
Hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
Hy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
H45 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], np.float32)
H135 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], np.float32)
gx = cv2.filter2D(filtrada, cv2.CV_32F, Hx)
gy = cv2.filter2D(filtrada, cv2.CV_32F, Hy)
g45 = cv2.filter2D(filtrada, cv2.CV_32F, H45)
g135 = cv2.filter2D(filtrada, cv2.CV_32F, H135)
magnitude = np.sqrt(gx**2 + gy**2 + g45**2 + g135**2)  # Equação (9).

# As equações (10) e (11) aparecem iguais no PDF. Para obter uma direção,
# projetamos geometricamente as respostas diagonais nos eixos x e y.
dx = (gx + (g45 + g135) / np.sqrt(2)).astype(np.float32)
dy = (gy + (g45 - g135) / np.sqrt(2)).astype(np.float32)

# 5. Afinar as bordas ANTES de aplicar Otsu, como descreve a seção III-C.
afinado = suprimir_nao_maximos(magnitude, dx, dy)
maximo = float(afinado.max())
if maximo > 0:
    # Otsu trabalha com 8 bits; depois devolvemos o limiar à escala original.
    mapa_8bits = np.uint8(np.clip(afinado * (255 / maximo), 0, 255))
    valor_otsu, _ = cv2.threshold(mapa_8bits, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    alto = max(1.0, float(valor_otsu) * maximo / 255)
    melhorado = ligar_bordas(afinado, alto / 2, alto)
else:
    alto = 0.0
    melhorado = np.zeros_like(original)

# 6. Salvar o painel e as etapas para inspeção.
saida = Path(__file__).with_name("resultados")
saida.mkdir(exist_ok=True)
cv2.imwrite(str(saida / "1_original.png"), original)
cv2.imwrite(str(saida / "2_ruido.png"), ruidosa)
cv2.imwrite(str(saida / "3_filtrada.png"), filtrada)
cv2.imwrite(str(saida / "4_canny_tradicional.png"), tradicional)
cv2.imwrite(str(saida / "5_canny_modificado.png"), melhorado)
painel = np.vstack((np.hstack((original, ruidosa)),
                    np.hstack((tradicional, melhorado))))
cv2.imwrite(str(saida / "comparacao.png"), painel)
print(f"Limiar alto (Otsu): {alto:.1f}; baixo: {alto / 2:.1f}")
print(f"Resultados: {saida}")
