# Secure-Mail

**Secure-Mail** est un assistant de rédaction d'emails, conçu pour garantir la confidentialité de vos données. Il fonctionne **100% en local** sur votre machine.

Ce projet utilise l'Intelligence Artificielle pour adapter vos brouillons d'emails aux règles de votre entreprise (ton, politesse, format) grâce à la technique du **RAG (Retrieval-Augmented Generation)**, tout en **anonymisant** automatiquement les informations sensibles avant de les traiter.

## Fonctionnalités Principales

- **100% Local & Privé :** Aucune donnée n'est envoyée dans le cloud. Tout le traitement (anonymisation et génération) se fait sur votre propre machine.
- **Anonymisation Automatique :** Détection et masquage des entités sensibles (Personnes, Organisations, Lieux) dans vos brouillons grâce à spaCy. Elles sont remplacées par la balise `[CONFIDENTIEL]`.
- **RAG (Génération Augmentée par la Recherche) :** Le modèle d'IA prend en compte des règles spécifiques, (ex : "Ton professionnel") stockées dans une base de données vectorielle locale.
- **Modèle Performant :** Utilisation du modèle LLM open-source **Mistral** via Ollama.
- **Interface Simple :** Une interface web claire et intuitive propulsée par Streamlit.

---

## Stack Technique

- **Interface Utilisateur :** [Streamlit](https://streamlit.io/)
- **Orchestration IA :** [LangChain](https://python.langchain.com/)
- **Modèle LLM Local :** [Ollama](https://ollama.com/) (modèle : `mistral`)
- **Base de Données Vectorielle :** [ChromaDB](https://www.trychroma.com/)
- **Embeddings :** HuggingFace (`all-MiniLM-L6-v2`) via `sentence-transformers`
- **NLP / Anonymisation :** [spaCy](https://spacy.io/) (modèle français : `fr_core_news_sm`)

---

## Prérequis et Installation

### 1. Prérequis système
- **Python 3.9+** installé sur votre machine.
- **Ollama** installé. Si ce n'est pas le cas, téléchargez-le sur [ollama.com](https://ollama.com/download) et installez-le.

### 2. Télécharger le modèle IA (Mistral)
Ouvrez un terminal et exécutez la commande suivante pour télécharger le modèle Mistral via Ollama (cette étape peut prendre quelques minutes selon votre connexion, le modèle pèse environ 4 Go) :
```bash
ollama pull mistral
```

### 3. Création d'un environnement virtuel et installation des dépendances
Il est **fortement conseillé** d'utiliser un environnement virtuel pour isoler les dépendances de ce projet et ne pas polluer votre installation Python globale.

Dans le dossier de votre projet (`secure-mail`), créez et activez le `venv` :

**Sur Mac/Linux :**
```bash
python -m venv venv
source venv/bin/activate
```

**Sur Windows :**
```bash
python -m venv venv
venv\Scripts\activate
```

Une fois l'environnement activé (vous devriez voir `(venv)` dans votre terminal), installez les bibliothèques :
```bash
pip install -r requirements.txt
```

### 4. Télécharger le modèle NLP pour l'anonymisation (spaCy)
Ce modèle permet à l'application de comprendre le français et de détecter les noms, lieux et entreprises :
```bash
python -m spacy download fr_core_news_sm
```

### 5. Initialiser la base de données de règles
Exécutez le script d'initialisation pour créer la base vectorielle locale (`./chroma_db`) qui contiendra les règles de votre entreprise :
```bash
python init_db.py
```
*Note : Vous pouvez modifier les règles directement dans le fichier `init_db.py` avant de l'exécuter si vous souhaitez les personnaliser.*

---

## Comment lancer l'application

1. Assurez-vous que l'application **Ollama** est bien lancée sur votre ordinateur (vérifiez qu'elle tourne en arrière-plan ou lancez `ollama serve` dans un terminal).
2. Ouvrez un terminal dans le dossier du projet (`secure-mail`).
3. Lancez l'interface web avec Streamlit :
```bash
streamlit run app.py
```
4. Votre navigateur va s'ouvrir automatiquement sur `http://localhost:8501`.
5. Saisissez votre brouillon d'email dans la zone de texte et cliquez sur **"Générer l'email"**.

---

## arrêter l'application

Pour tout fermer proprement, suivez ces deux étapes :

1. **Arrêter l'interface web (Streamlit) :**
   Retournez dans le terminal où vous avez tapé `streamlit run app.py` et appuyez simultanément sur les touches :
   `Ctrl + C`
   *(Cela coupe le serveur local).*

2. **Arrêter l'IA (Ollama) :**
   - Si vous utilisez l'application bureau (ex: sur Mac, le petit lama dans la barre des menus en haut à droite), cliquez dessus et sélectionnez **"Quit Ollama"**.
   - Si vous avez lancé `ollama serve` dans un terminal, retournez dans ce terminal et faites `Ctrl + C`.

3. **Fermer la page web :**
   Vous pouvez simplement fermer l'onglet dans votre navigateur (Chrome, Safari, etc.).

---

## Structure du Projet

```text
secure-mail/
│
├── app.py               # Fichier principal : Interface Streamlit, Anonymisation et RAG
├── init_db.py           # Script pour initialiser ChromaDB avec les règles d'entreprise
├── requirements.txt     # Liste des dépendances Python
├── README.md            # Ce fichier de documentation
│
└── chroma_db/           # Dossier auto-généré (par init_db.py) contenant la base de données 
```

