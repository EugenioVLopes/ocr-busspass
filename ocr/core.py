import io
import json
import logging
import os

import cv2
import fitz  # Importe a biblioteca PyMuPDF
import numpy as np
import pytesseract
from PIL import Image

# Configuração do caminho para o executável do Tesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_image(image, threshold=140, use_grayscale=True, use_binarization=True, use_median_blur=False, blur_kernel_size=5):
    """
    Pré-processa a imagem para melhorar a precisão do OCR.

    Args:
        image (PIL.Image): Objeto de imagem PIL.
        threshold (int): Valor de limiar para binarização (default: 140).
        use_grayscale (bool): Se True, converte a imagem para escala de cinza (default: True).
        use_binarization (bool): Se True, binariza a imagem (default: True).
        use_median_blur (bool): Se True, aplica filtro de mediana para remover ruído (default: False).
        blur_kernel_size (int): Tamanho do kernel para o filtro de mediana (default: 5).

    Returns:
        PIL.Image: Imagem pré-processada.
    """
    logger.info("Iniciando pré-processamento da imagem.")

    if use_grayscale:
        image = image.convert('L')
        logger.debug("Imagem convertida para escala de cinza.")

    if use_binarization:
        image = image.point(lambda x: 0 if x < threshold else 255, '1')
        logger.debug(f"Imagem binarizada com limiar {threshold}.")

    if use_median_blur:
        image_np = np.array(image)
        image_np = cv2.medianBlur(image_np, blur_kernel_size)
        image = Image.fromarray(image_np)
        logger.debug(f"Filtro de mediana aplicado com tamanho de kernel {blur_kernel_size}.")

    logger.info("Pré-processamento da imagem concluído.")
    return image


def extract_text_from_image(image_path, threshold=140, oem=3, psm=6, lang='por', **kwargs):
    """
    Extrai texto de uma imagem usando Tesseract OCR.

    Args:
        image_path (str): Caminho para a imagem.
        threshold (int): Valor de limiar para binarização (default: 140).
        oem (int): Modo de reconhecimento de caracteres do Tesseract (default: 3).
        psm (int): Modo de segmentação de página do Tesseract (default: 6).
        lang (str): Idioma para o OCR (default: 'por' para português).
        **kwargs: Argumentos adicionais para preprocess_image.

    Returns:
        str: Texto extraído da imagem.
    """
    logger.info(f"Iniciando extração de texto da imagem: {image_path}")
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Arquivo de imagem não encontrado: {image_path}")

        image = Image.open(image_path)
        logger.info("Imagem aberta com sucesso.")

        processed_image = preprocess_image(image, threshold, **kwargs)
        logger.info("Imagem pré-processada com sucesso.")

        custom_config = f'--oem {oem} --psm {psm} -l {lang}'
        text = pytesseract.image_to_string(processed_image, config=custom_config)

        if not text:
            logger.warning("Nenhum texto extraído da imagem.")

        logger.info(f"Extração de texto da imagem concluída: {image_path}")
        return text

    except FileNotFoundError as e:
        logger.error(str(e))
        return None
    except Exception as e:
        logger.error(f"Erro ao processar imagem: {e}", exc_info=True)
        return None


def extract_text_from_pdf(pdf_path):
    """
    Extrai texto de um PDF convertendo cada página em imagem e aplicando OCR.

    Args:
        pdf_path (str): Caminho para o arquivo PDF.

    Returns:
        str: Texto extraído do PDF.
    """
    logger.info(f"Iniciando extração de texto do PDF (convertendo para imagens): {pdf_path}")
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

        doc = fitz.open(pdf_path)
        all_text = ""

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Salve a imagem temporariamente
            temp_img_path = f"temp_page_{page_num}.png"
            img.save(temp_img_path)

            # Extraia o texto da imagem
            page_text = extract_text_from_image(temp_img_path)
            all_text += page_text if page_text else ""

            # Remova a imagem temporária
            os.remove(temp_img_path)

        logger.info(f"Extração de texto do PDF (convertendo para imagens) concluída: {pdf_path}")
        return all_text

    except FileNotFoundError as e:
        logger.error(str(e))
        return None
    except Exception as e:
        logger.error(f"Erro ao processar PDF: {e}", exc_info=True)
        return None

def extract_text_from_pdf_by_image(pdf_path, dpi=300):
    """
    Extrai texto de um PDF convertendo cada página em imagem e aplicando OCR com Tesseract.

    Args:
        pdf_path (str): Caminho para o arquivo PDF.
        dpi (int): Resolução em DPI para a conversão de PDF para imagem (default: 300).

    Returns:
        str: Texto extraído do PDF.
    """
    logger.info(f"Iniciando extração de texto do PDF (convertendo para imagens): {pdf_path}")
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

        doc = fitz.open(pdf_path)
        all_text = ""

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)  # Use dpi para controlar a resolução
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Salve a imagem temporariamente (opcional, útil para depuração)
            temp_img_path = f"temp_page_{page_num}.png"
            img.save(temp_img_path)

            # Pré-processamento da imagem (opcional, ajuste conforme necessário)
            processed_image = preprocess_image(img, threshold=150, use_grayscale=True, use_binarization=True, use_median_blur=False)

            # Salve a imagem pré-processada temporariamente (opcional, útil para depuração)
            temp_processed_img_path = f"temp_processed_page_{page_num}.png"
            processed_image.save(temp_processed_img_path)


            # Extraia o texto da imagem com Tesseract
            # Experimente diferentes configurações do Tesseract aqui, se necessário
            page_text = pytesseract.image_to_string(processed_image, config=f'--oem 3 --psm 6 -l por')
            all_text += page_text if page_text else ""

            # Remova as imagens temporárias (opcional)
            os.remove(temp_img_path)
            os.remove(temp_processed_img_path)

        logger.info(f"Extração de texto do PDF (convertendo para imagens) concluída: {pdf_path}")
        return all_text

    except FileNotFoundError as e:
        logger.error(str(e))
        return None
    except Exception as e:
        logger.error(f"Erro ao processar PDF: {e}", exc_info=True)
        return None

def convert_text_to_json(text):
    """
    Converte o texto extraído para um formato JSON.

    Args:
        text (str): Texto a ser convertido.

    Returns:
        str: String representando o JSON.
    """
    try:
        data = {"text": text}
        return json.dumps(data, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Erro ao converter texto para JSON: {e}", exc_info=True)
        return json.dumps({"error": str(e)})