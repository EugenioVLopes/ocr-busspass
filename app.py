from flask import Flask, jsonify
from api.routes import api_blueprint
from werkzeug.exceptions import HTTPException
import logging

app = Flask(__name__)

app.register_blueprint(api_blueprint, url_prefix='/api')

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,  # Nível mínimo de log (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Salva os logs em um arquivo chamado app.log
        logging.StreamHandler()  # Exibe os logs no console também
    ]
)

#Tratamento de erro centralizado 
@app.errorhandler(HTTPException)
def error_handler(e):
    """Retorna JSON em vez de HTML para erros HTTP."""
    return jsonify({
        "codigo": e.code,
        "nome": e.name,
        "descricao": e.description,
    })

@app.errorhandler
def handler_unexpected_error(e):
    """Trata erros inesperados."""
    return jsonify({
        "codigo": 500, 
        "nome": "Erro interno do servidor", 
        "descricao": "Ocorreu um erro inesperado"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)