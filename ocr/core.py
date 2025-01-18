import io
import json
import logging
import os

import cv2
import fitz  # Importe a biblioteca PyMuPDF
import numpy as np
import pytesseract
from PIL import Image
from functools import lru_cache

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_image(image, threshold=140, use_grayscale=True, use_binarization=True, 
                    use_median_blur=False, blur_kernel_size=3, use_denoising=False, **kwargs):
    """
    Pré-processa a imagem para melhorar a precisão do OCR.
    """
    try:
        # Validação de entrada
        if not isinstance(image, Image.Image):
            raise ValueError("Input deve ser um objeto PIL.Image")

        # Debug logs apenas se necessário
        if logger.level == logging.DEBUG:
            image.save('debug_original.png')
            logger.debug("Iniciando pré-processamento com parâmetros: threshold=%d, grayscale=%s, binarization=%s",
                      threshold, use_grayscale, use_binarization)

        if use_grayscale:
            image = image.convert('L')

        if use_binarization:
            image = image.point(lambda x: 0 if x < threshold else 255, '1')

        if use_median_blur:
            image_np = np.array(image)
            image_np = cv2.medianBlur(image_np, blur_kernel_size)
            image = Image.fromarray(image_np)

        if use_denoising:
            image_np = np.array(image)
            image_np = cv2.fastNlMeansDenoising(image_np, None, 30, 7, 21)
            image = Image.fromarray(image_np)

        # Debug logs apenas se necessário
        if logger.level == logging.DEBUG:
            image.save('debug_processed.png')
            logger.debug("Pré-processamento concluído com sucesso")

        return image

    except Exception as e:
        logger.error("Erro no pré-processamento: %s", str(e), exc_info=True)
        raise


@lru_cache(maxsize=100)
def extract_text_from_image(image_path, threshold=140, oem=3, psm=6, lang='por', **kwargs):
    """
    Versão cacheada da extração de texto
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Arquivo de imagem não encontrado: {image_path}")

        image = Image.open(image_path)
        processed_image = preprocess_image(image, threshold, **kwargs)
        
        custom_config = f'--oem {oem} --psm {psm} -l {lang}'
        text = pytesseract.image_to_string(processed_image, config=custom_config)

        if not text:
            logger.warning("Nenhum texto extraído da imagem: %s", image_path)
            return ""

        return text

    except FileNotFoundError as e:
        logger.error(str(e))
        return None
    except Exception as e:
        logger.error("Erro ao processar imagem %s: %s", image_path, str(e), exc_info=True)
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

def extract_text_from_pdf_by_image(pdf_path, dpi=300, lang='por', **kwargs):
    """
    Extrai texto de um PDF convertendo cada página em imagem e aplicando OCR.
    """
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

        logger.info("Processando PDF: %s (dpi=%d, lang=%s)", pdf_path, dpi, lang)
        doc = fitz.open(pdf_path)
        all_text = []

        for page_num in range(doc.page_count):
            logger.debug("Processando página %d/%d", page_num + 1, doc.page_count)
            
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            kwargs.pop('dpi', None)
            processed_image = preprocess_image(img, **kwargs)
            
            page_text = pytesseract.image_to_string(processed_image, lang=lang)
            if page_text:
                all_text.append(page_text.strip())
            else:
                logger.warning("Nenhum texto extraído da página %d", page_num + 1)

        if not all_text:
            logger.warning("Nenhum texto extraído do PDF")
            return None

        return "\n".join(all_text)

    except Exception as e:
        logger.error("Erro ao processar PDF %s: %s", pdf_path, str(e), exc_info=True)
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
        return json.dumps({"error": str(e)}, ensure_ascii=False)

class OCRManager:
    def __init__(self, tesseract_path=None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        self.supported_languages = {'por', 'eng', 'spa'}
        self._initialize_tesseract()
        logger.info("OCRManager inicializado com sucesso")
    
    def _initialize_tesseract(self):
        try:
            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract inicializado (versão %s)", version)
        except Exception as e:
            logger.error("Erro ao inicializar Tesseract: %s", str(e))
            raise RuntimeError("Tesseract não está configurado corretamente")
    
    def process_document(self, file_path, language='por', dpi=300, **kwargs):
        """
        Processa um documento (PDF ou imagem)
        """
        try:
            if language not in self.supported_languages:
                raise ValueError(f"Idioma não suportado: {language}")
                
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info("Processando documento: %s (tipo=%s, lang=%s)", file_path, file_ext, language)
            
            if file_ext == '.pdf':
                return self._process_pdf(file_path, language=language, dpi=dpi)
            elif file_ext in ['.png', '.jpg', '.jpeg']:
                kwargs.pop('dpi', None)
                return self._process_image(file_path, language=language, **kwargs)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {file_ext}")
                
        except Exception as e:
            logger.error("Erro ao processar documento %s: %s", file_path, str(e))
            raise
    
    def _process_pdf(self, pdf_path, language, dpi=300, **kwargs):
        """Processa arquivo PDF"""
        return extract_text_from_pdf_by_image(pdf_path, lang=language, dpi=dpi)
    
    def _process_image(self, image_path, language, **kwargs):
        """Processa arquivo de imagem"""
        return extract_text_from_image(image_path, lang=language, **kwargs)