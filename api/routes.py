from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
from ocr.core import extract_text_from_image, extract_text_from_pdf
from ocr.utils import allowed_file
import logging

api_blueprint = Blueprint('api', __name__)

POPPLER_PATH = r'C:\Program Files\poppler-24.08.0'

logger = logging.getLogger(__name__)


@api_blueprint.route('/ocr', methods=['POST'])
def ocr_route():
    """Rota para processar imagens e extrair texto."""
    logger.info("--------------------------------------------")
    logger.info("Requisição recebida em /ocr")

    if 'file' not in request.files:
        logger.warning("Nenhum arquivo enviado na requisição.")
        return jsonify({"mensagem": "Nenhum arquivo enviado"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"mensagem": "Nenhum arquivo selecionado"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.root_path, 'uploads', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        logger.info(f"Arquivo salvo: {filepath}")
        
        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_image(filepath)
        
        # Remove o arquivo temporário após a extração do texto
        os.remove(filepath)
        logger.info(f"Arquivo temporário removido: {filepath}")
        
        if text:
            logger.info("Texto extraído com sucesso.")
            return jsonify({"texto": text})
        else:
            logger.error("Falha ao extrair texto do arquivo.")
            return jsonify({"mensagem": "Erro ao extrair texto da imagem"}), 500
    else:
        logger.warning(f"Tipo de arquivo não permitido: {file.filename}")
        return jsonify({"mensagem": "Tipo de arquivo não suportado"}), 400