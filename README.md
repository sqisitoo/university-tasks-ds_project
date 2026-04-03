# Detekcja Katastrof w Mediach Społecznościowych przy użyciu LLM

## 📌 Opis i Cel Projektu
Głównym celem projektu jest stworzenie, analiza i dostrojenie (fine-tuning) modelu językowego (LLM) z biblioteki HuggingFace Transformers do automatycznej klasyfikacji krótkich form tekstowych. 

Aplikacja ma za zadanie analizować posty z serwisu X (dawniej Twitter) i z wysoką skutecznością odróżniać tweety informujące o rzeczywistych katastrofach (np. pożarach, trzęsieniach ziemi, wypadkach) od postów, które wykorzystują słownictwo związane z zagrożeniami w kontekście potocznym lub metaforycznym.

## 📊 Źródło i Opis Danych
Wykorzystany zbiór danych to **"Natural Language Processing with Disaster Tweets"** pochodzący z ogólnodostępnej platformy Kaggle.
- **Źródło:** [Kaggle Dataset Link](https://www.kaggle.com/c/nlp-getting-started/data)
- **Rozmiar zbioru:** Ponad 7500 unikalnych tweetów (wystarczający do procesu fine-tuningu na późniejszych etapach).
- **Zmienna objaśniana (Target):** Klasyfikacja binarna. `1` oznacza rzeczywistą katastrofę, `0` oznacza brak zagrożenia.
- **Zmienne objaśniające (Features):** Głównym nośnikiem informacji jest kolumna `text` (treść tweeta). Kolumny poboczne to `keyword` oraz `location`.

## 📂 Struktura Repozytorium
Repozytorium zostało zorganizowane zgodnie z dobrymi praktykami inżynierii danych i data science:

```text
./
│
├── README.md               # Dokumentacja główna projektu
├── raport.pdf              # raport PDF dla Etapu 2
├── requirements.txt        # Lista zależności i bibliotek Pythona
│
├── notebooks/                    # Kody źródłowe i notatniki
│   └── 01_EDA.ipynb        # Notatnik Jupyter z główną analizą i czyszczeniem danych
│
├── data/                   # Katalog na surowe dane (nieśledzony, jeśli dane są duże)
│   └── train.csv           # Pobrany zbiór treningowy z Kaggle
│
└── outputs/                # Zapisane wyniki działania skryptów i modele
    ├── plot1.png           # Wygenerowane wykresy (np. WordCloud)
    └── train_cleaned.csv   # Oczyszczony zbiór danych gotowy dla modelu LLM
