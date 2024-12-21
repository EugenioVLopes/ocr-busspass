from PIL import Image
import pytesseract
import fitz  
import io
import logging
import os
from .utils import allowed_file
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Obtém um logger para este módulo
logger = logging.getLogger(__name__)

def preprocess_image(image):
    """Pré-processa a imagem para melhorar a precisão do OCR.
    Args:
        image (PIL.Image): Objeto de imagem PIL.
    Returns:
        PIL.Image: Imagem pré-processada.
    """
    logger.info("Iniciando pré-processamento da imagem.")
    # Converte a imagem para escala de cinza
    image = image.convert('L')
    # Binariza a imagem
    image = image.point(lambda x: 0 if x < 140 else 255, '1')
    logger.info("Pré-processamento da imagem concluído.")
    return image

def extract_text_from_image(image_path):
    """Extrai texto de uma imagem usando Tesseract OCR.
    Args:
        image_path (str): Caminho para a imagem.
    Returns:
        str: Texto extraído da imagem.
    """
    logger.info(f"Iniciando extração de texto da imagem: {image_path}")
    try:
        image = Image.open(image_path)
        logger.info("Imagem aberta com sucesso.")
        processed_image = preprocess_image(image)
        logger.info("Imagem pré-processada com sucesso.")
        # Ajuste das configurações do Tesseract
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        logger.info(f"Extração de texto da imagem concluída: {image_path}")
        return text
    
    except Exception as e:
        logger.error(f"Erro ao processar imagem: {e}")
        return None
    
def extract_text_from_pdf(pdf_path):
    """Extrai imagens de um PDF e as processa com OCR, retornando o texto extraído."""
    logger.info(f"Iniciando extração de texto do PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        all_text = ""
        for page in doc:
            image_list = page.get_images(full=True)
            logger.info(f"Processando página com {len(image_list)} imagens.")
            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Converte bytes para uma imagem PIL
                image = Image.open(io.BytesIO(image_bytes))
                logger.info("Imagem extraída do PDF com sucesso.")

                # Processa a imagem e extrai o texto
                processed_image = preprocess_image(image)
                logger.info("Imagem pré-processada com sucesso.")
                # Ajusta as configurações do Tesseract
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(processed_image, config=custom_config)
                all_text += text + "\n"  # Adiciona uma quebra de linha entre as imagens
                logger.info(f"Texto extraído da imagem XREF {xref} na página.")
        logger.info("Extração de texto do PDF concluída.")
        return all_text
    
    except Exception as e:
        logger.error(f"Erro ao processar PDF: {e}")
        return None

# def extract_text_from_pdf(pdf_path, poppler_path=None):
#     """Converte um PDF para imagens e as processa com OCR, retornando o texto extraído."""
#     images = convert_from_path(pdf_path, poppler_path=poppler_path)
#     all_text = ""
#     for image in images:
#         processed_image = preprocess_image(image)
#         text = pytesseract.image_to_string(processed_image)
#         all_text += text + "\n"  # Adiciona uma quebra de linha entre as páginas
#     return all_text
