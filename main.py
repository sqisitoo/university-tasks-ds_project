# main.py
import gradio as gr
from transformers import pipeline
import os

def load_model():
    model_path = "./disaster_model"
    
    # Sprawdzenie, czy użytkownik pobrał wytrenowany model
    if not os.path.exists(model_path):
        print(f"⚠️ UWAGA: Nie znaleziono wytrenowanego modelu w {model_path}.")
        print("Aby aplikacja działała z fine-tuned modelem, uruchom najpierw notatnik z Etapu 4.")
        print("Ładowanie bazowego modelu Zero-Shot jako zastępstwo...\n")
        # Powrót do etapu 3, jeśli brak modelu
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    print("✅ Znaleziono wytrenowany model. Ładowanie...")
    return pipeline("text-classification", model=model_path, tokenizer=model_path)

def predict(text):
    classifier = load_model()
    
    # Jeśli załadował się Zero-Shot (brak modelu lokalnie)
    if classifier.task == "zero-shot-classification":
        result = classifier(text, candidate_labels=["real disaster", "not a disaster"])
        label = "🔥 PRAWDZIWA KATASTROFA" if result['labels'][0] == "real disaster" else "✅ Zwykły tweet"
        score = result['scores'][0]
    # Jeśli załadował się nasz model Fine-Tuned
    else:
        result = classifier(text)[0]
        label = "🔥 PRAWDZIWA KATASTROFA" if result['label'] == 'LABEL_1' else "✅ Zwykły tweet (Brak zagrożenia)"
        score = result['score']
        
    return f"{label} (Pewność: {score*100:.2f}%)"

def main():
    print("Uruchamianie aplikacji internetowej...")
    
    # Interfejs użytkownika
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=3, placeholder="Wpisz treść tweeta tutaj (np. 'Forest fire near the city!')..."),
        outputs=gr.Textbox(label="Wynik detekcji"),
        title="🚨 Detektor Katastrof (LLM)",
        description="Aplikacja wykorzystująca model językowy z HuggingFace do klasyfikacji tweetów.",
        theme="default"
    )
    
    # Uruchomienie serwera WWW na localhost
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()