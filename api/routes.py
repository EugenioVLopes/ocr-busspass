import logging
import os

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from ocr.core import OCRManager
from ocr.utils import allowed_file

api_blueprint = Blueprint('api', __name__)

logger = logging.getLogger(__name__)

ocr_manager = OCRManager()

@api_blueprint.route('/ocr', methods=['POST'])
def ocr_route():
    """Rota para processar imagens e extrair texto."""
    try:
        # Validar arquivo
        if 'file' not in request.files:
            logger.warning("Requisição sem arquivo")
            return jsonify({"erro": "Nenhum arquivo enviado"}), 400

        file = request.files['file']
        if not file or file.filename == '':
            logger.warning("Arquivo inválido ou vazio")
            return jsonify({"erro": "Arquivo inválido"}), 400

        if not allowed_file(file.filename):
            logger.warning("Tipo de arquivo não permitido: %s", file.filename)
            return jsonify({"erro": "Tipo de arquivo não suportado"}), 400

        # Parâmetros opcionais com validação
        try:
            language = request.form.get('language', 'por')
            dpi = int(request.form.get('dpi', 300))
            if dpi < 72 or dpi > 600:
                logger.warning("DPI inválido: %d", dpi)
                return jsonify({"erro": "DPI deve estar entre 72 e 600"}), 400
        except ValueError:
            logger.warning("Valor de DPI inválido na requisição")
            return jsonify({"erro": "Valor de DPI inválido"}), 400
        
        # Processar arquivo
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.root_path, 'uploads', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            file.save(filepath)
            logger.info("Processando arquivo: %s (lang=%s, dpi=%d)", filename, language, dpi)
            
            text = ocr_manager.process_document(
                filepath, 
                language=language,
                dpi=dpi
            )
            
            if not text:
                logger.warning("Nenhum texto extraído do arquivo: %s", filename)
                return jsonify({"erro": "Não foi possível extrair texto"}), 422
                
            cleaned_text = text.replace("\n", " ").replace("\r", " ").strip()
            logger.info("Arquivo processado com sucesso: %s", filename)
            return jsonify({"text": cleaned_text})
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug("Arquivo temporário removido: %s", filepath)

    except Exception as e:
        logger.error("Erro no processamento: %s", str(e), exc_info=True)
        return jsonify({"erro": "Erro interno do servidor"}), 500