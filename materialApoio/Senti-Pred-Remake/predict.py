import joblib
import re
import os

# Caminhos dos arquivos salvos pelo train_god_mode.py (O recordista de 97.5%)
MODEL_PATH = "god_mode_model.pkl"
VECTORIZER_PATH = "god_mode_vectorizer.pkl"

def clean_text_god_mode(text):
    text = str(text).lower()
    # Remover URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remover caracteres repetidos (ex: "loooove" -> "love")
    text = re.sub(r'(.)\1+', r'\1\1', text)
    # Mantendo apenas letras, espaços e pontuação emocional importante (!, ?)
    text = re.sub(r'[^a-z\s\!\?]', '', text)
    return " ".join(text.split())

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print(f"Erro: Arquivos do modelo '{MODEL_PATH}' não encontrados.")
        print("Certifique-se de ter rodado o 'train_god_mode.py' primeiro.")
        return None, None
    
    print(f"--- Carregando o Modelo God Mode (97.5% Acc) ---")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    cleaned = clean_text_god_mode(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return prediction

def main():
    model, vectorizer = load_model()
    
    if model and vectorizer:
        print("\n🚀 Modelo de Elite carregado! Digite suas frases (em inglês) para análise.")
        print("Digite 'sair' para encerrar.")
        
        while True:
            user_input = input("\n📝 Tweet: ")
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando... Parabéns pelo recorde de 97.5%!")
                break
            
            if not user_input.strip():
                continue
                
            sentiment = predict_sentiment(user_input, model, vectorizer)
            
            # Formatação visual
            emoji = ""
            if sentiment == "Positive": emoji = "🟢"
            elif sentiment == "Negative": emoji = "🔴"
            elif sentiment == "Neutral": emoji = "⚪"
            else: emoji = "🟡" # Irrelevant
            
            print(f"📊 Sentimento: {emoji} {sentiment}")

if __name__ == "__main__":
    main()
