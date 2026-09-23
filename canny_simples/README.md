# Canny: versão simples para começar

Este primeiro experimento tem um único script comentado e uma imagem de exemplo. Ele mostra as etapas iniciais do artigo de Li e Liu (2022) sem apresentar toda a matemática de uma vez.

## Como executar

Abra o terminal na pasta `canny_simples` e digite:

```powershell
python -m pip install -r requirements.txt
python canny_simples.py
```

Abra `resultados/comparacao.png`. No painel, a linha de cima contém **imagem original | imagem com ruído**; a linha de baixo contém **Canny tradicional | Canny com mediana e Otsu**. Cada imagem também é salva separadamente.

Para usar outra imagem, altere a linha `imagem = Path(__file__).with_name("imagem_exemplo.png")` para, por exemplo, `imagem = Path(r"C:\Users\Pedro\Imagens\lena.png")`.

## O que cada parte faz?

1. `imread`: abre a imagem em tons de cinza.
2. `choice`: seleciona 2% das posições; metade vira preto e metade vira branco.
3. `GaussianBlur` e `Canny`: produzem a comparação tradicional.
4. `medianBlur`: remove muitos pontos isolados de sal e pimenta.
5. `Sobel` mede a intensidade das mudanças. `threshold(... OTSU)` escolhe o limiar alto e o baixo é sua metade.
6. `imwrite`: grava os resultados para comparar visualmente.

**Escopo desta primeira etapa:** o filtro de mediana é uma aproximação simples do filtro descrito no artigo. As máscaras adicionais de **45° e 135° ainda não foram implementadas neste arquivo**; elas podem ser acrescentadas depois de entender essas seis etapas. Não use os resultados como reprodução numérica da Tabela I do artigo.

Artigo: Yibo Li e Bailun Liu, *Improved edge detection algorithm for canny operator*, ITAIC, 2022, DOI: 10.1109/ITAIC54216.2022.9836608.
