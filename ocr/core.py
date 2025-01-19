import io
import json
import logging
import os

import cv2
import fitz
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

    def process_document_with_position(self, file_path, language='por', **kwargs):
        """
        Processa documento e retorna campos identificados
        """
        try:
            if language not in self.supported_languages:
                raise ValueError(f"Idioma não suportado: {language}")
                
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info("Processando documento: %s", file_path)
            
            if file_ext == '.pdf':
                pages = self._process_pdf_with_position(file_path, language=language, **kwargs)
                if pages:
                    # Combinar campos de todas as páginas
                    all_fields = {}
                    for page in pages:
                        all_fields.update(page.get('fields', {}))
                    return all_fields
            elif file_ext in ['.png', '.jpg', '.jpeg']:
                return extract_text_with_position(file_path, lang=language, **kwargs)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {file_ext}")
                
        except Exception as e:
            logger.error("Erro ao processar documento: %s", str(e))
            raise
    
    def _process_pdf_with_position(self, pdf_path, language, **kwargs):
        """Processa PDF retornando texto com posicionamento de cada página"""
        try:
            doc = fitz.open(pdf_path)
            results = []
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Salva temporariamente para processamento
                temp_path = f"temp_page_{page_num}.png"
                img.save(temp_path)
                
                try:
                    page_result = extract_text_with_position(temp_path, lang=language)
                    if page_result:
                        results.append({
                            'page': page_num + 1,
                            'elements': page_result
                        })
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            return results
            
        except Exception as e:
            logger.error("Erro ao processar PDF com posição: %s", str(e))
            return None

def extract_text_with_position(image_path, lang='por', **kwargs):
    """
    Extrai texto com informações de posicionamento e retorna campos identificados.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")

        # Extrair dados usando Tesseract com formato DICT
        data = pytesseract.image_to_data(
            Image.open(image_path), 
            lang=lang, 
            output_type=pytesseract.Output.DICT
        )
        
        # Organizar resultado
        elements = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) == -1 or not data['text'][i].strip():
                continue
                
            left = int(data['left'][i])
            top = int(data['top'][i])
            width = int(data['width'][i])
            height = int(data['height'][i])
            
            elements.append({
                'text': data['text'][i],
                'confidence': float(data['conf'][i]),
                'position': {
                    'x': left + (width // 2),
                    'y': top + (height // 2),
                    'width': width,
                    'height': height,
                    'left': left,
                    'top': top
                }
            })
        
        # Identificar e retornar apenas os campos encontrados
        fields = identify_fields(elements)
        return {k: v['value'] for k, v in fields.items()}

    except Exception as e:
        logger.error("Erro ao extrair texto com posição: %s", str(e), exc_info=True)
        return None

def identify_fields(elements):
    """
    Identifica campos baseados no texto e posição dos elementos.
    """
    fields = {}
    field_patterns = {
        'nome': ['nome', 'name'],
        'cpf': ['cpf', 'cadastro de pessoa'],
        'rg': ['rg', 'registro geral', 'identidade'],
        'data_nascimento': ['nascimento', 'data de nascimento', 'birth', 'nasc'],
        'filiacao': ['filiacao', 'filho', 'pai', 'mae', 'father', 'mother'],
        'endereco': ['endereco', 'address', 'residencia'],
        'validade': ['validade', 'valid', 'expira'],
        'numero_cartao': ['cartão', 'card number', 'número do cartão'],
        'validade_cartao': ['valid thru', 'validade'],
        'titular': ['titular', 'holder']
    }

    # Ordenar elementos por posição vertical (y)
    sorted_elements = sorted(elements, key=lambda x: x['position']['y'])
    
    for i, element in enumerate(sorted_elements):
        text = element['text'].lower().strip()
        
        # Procurar por labels de campos
        for field_name, patterns in field_patterns.items():
            if any(pattern in text for pattern in patterns):
                # Procurar valor do campo nas proximidades
                value = find_field_value(sorted_elements, i, element['position'])
                if value:
                    fields[field_name] = {
                        'label': element['text'],
                        'value': value['text'],
                        'confidence': value['confidence'],
                        'position': value['position']
                    }
                break
    
    return fields

def find_field_value(elements, current_index, label_position, max_distance=100):
    """
    Encontra o valor de um campo baseado na posição do label.
    """
    label_x = label_position['x']
    label_y = label_position['y']
    
    # Procurar primeiro à direita (mesmo y, x maior)
    for element in elements[current_index:]:
        pos = element['position']
        
        # Verificar se está na mesma linha (aproximadamente)
        if abs(pos['y'] - label_y) < 20:
            if pos['x'] > label_x:
                return element
    
    # Se não encontrou à direita, procurar abaixo
    for element in elements[current_index:]:
        pos = element['position']
        
        # Verificar se está abaixo e próximo
        if (pos['y'] > label_y and 
            pos['y'] - label_y < max_distance and 
            abs(pos['x'] - label_x) < max_distance):
            return element
    
    return None