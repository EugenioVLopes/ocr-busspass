import requests
import json
import os
from difflib import SequenceMatcher
import time

def extract_ground_truth(words_file, form_id):
  """Extrai o ground truth (texto correto) de um formulário."""
  ground_truth = ""
  with open(words_file, "r") as f:
    for line in f:
      parts = line.strip().split()
      if parts[0] == form_id:
        ground_truth += parts[-1] + " "
  return ground_truth.strip()

def process_image(image_path, filename):
    """Processa uma única imagem através da API OCR"""
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            response = requests.post("http://127.0.0.1:8002/api/ocr", files=files)
            
            if response.status_code != 200:
                raise Exception(f"Erro HTTP {response.status_code}")

            # Remove escape duplo e decodifica
            text = response.text.strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]  # Remove aspas externas
            
            # Decodifica caracteres de escape
            text = text.encode().decode('unicode_escape')
            
            # Parse JSON
            result = json.loads(text)
            
            # Valida estrutura
            if not isinstance(result, dict) or "text" not in result:
                raise ValueError(f"Formato inválido: {result}")
                
            return result
            
    except Exception as e:
        print(f"Erro ao processar {filename}: {e}")
        return None

def test_ocr_api():
    """Testa a API de OCR com as imagens do formsA-D"""
    # Métricas
    total_docs = 0
    successful_docs = 0
    failed_docs = 0
    similarity_scores = []
    start_time = time.time()

    # Configuração de diretórios
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "data", "words","a01","a01-000u")
    words_file = os.path.join(base_dir, "data", "words.txt")

    # Verificação de arquivos
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Diretório não encontrado: {image_dir}")
    if not os.path.exists(words_file):
        raise FileNotFoundError(f"Arquivo ground truth não encontrado: {words_file}")

    # Verificação de arquivos dentro do diretório
    if not any(fname.endswith(".png") for fname in os.listdir(image_dir)):
        raise FileNotFoundError(f"Nenhum arquivo .png encontrado no diretório: {image_dir}")

    for filename in os.listdir(image_dir):
        if not filename.endswith(".png"):
            continue
            
        total_docs += 1
        image_path = os.path.join(image_dir, filename)
        form_id = filename[:-4]

        result = process_image(image_path, filename)
        if result is None:
            failed_docs += 1
            continue

        ground_truth = extract_ground_truth(words_file, form_id)
        if ground_truth:
            try:
                similarity = SequenceMatcher(None, result["text"], ground_truth).ratio()
                similarity_scores.append(similarity)
                successful_docs += 1
            except Exception as e:
                failed_docs += 1
                print(f"Erro ao calcular similaridade: {e}")

    # Relatório final
    print("\n=== Relatório de Processamento OCR ===")
    print(f"Total de documentos: {total_docs}")
    print(f"Documentos processados com sucesso: {successful_docs}")
    print(f"Falhas: {failed_docs}")
    
    if similarity_scores:
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        print(f"Similaridade média: {avg_similarity:.2%}")
        print(f"Melhor similaridade: {max(similarity_scores):.2%}")
        print(f"Pior similaridade: {min(similarity_scores):.2%}")
    
    total_time = time.time() - start_time
    print(f"Tempo total de processamento: {total_time:.2f} segundos")
    print("=====================================")

if __name__ == "__main__":
    try:
        test_ocr_api()
    except Exception as e:
        print(f"Erro fatal: {e}")