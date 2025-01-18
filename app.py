from flask import Flask, jsonify
from api.routes import api_blueprint
from werkzeug.exceptions import HTTPException
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

app.register_blueprint(api_blueprint, url_prefix='/api')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

@app.errorhandler(HTTPException)
def error_handler(e):
    """Retorna JSON em vez de HTML para erros HTTP."""
    return jsonify({
        "codigo": e.code,
        "nome": e.name,
        "descricao": e.description,
    })

@app.errorhandler(Exception)
def handler_unexpected_error(e):
    return jsonify({
        "codigo": 500, 
        "nome": "Erro interno do servidor", 
        "descricao": "Ocorreu um erro inesperado"
    }), 500

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)