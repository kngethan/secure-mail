import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

RULES_FILE = "rules.json"

def get_current_rules():
    """Charge les règles actuelles depuis le fichier JSON."""
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data.get("rules", [])
            except json.JSONDecodeError:
                pass
    
    # Règles par défaut si le fichier n'existe pas ou est corrompu
    return [
        "Règle 1 : Adapte le ton au destinataire (tutoiement ou vouvoiement selon le contexte).",
        "Règle 2 : Le ton doit toujours rester courtois et respectueux.",
        "Règle 3 : Le style d'écriture doit être clair, concis et aller droit au but."
    ]

def init_database(custom_rules=None):
    print("Initialisation de la base vectorielle ChromaDB...")
    
    # Utiliser les règles personnalisées fournies, ou charger les existantes
    if custom_rules is not None:
        rules = custom_rules
        # Sauvegarder les nouvelles règles
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            json.dump({"rules": rules}, f, ensure_ascii=False, indent=4)
    else:
        rules = get_current_rules()
        # Créer le fichier avec les règles par défaut si nécessaire
        if not os.path.exists(RULES_FILE):
             with open(RULES_FILE, "w", encoding="utf-8") as f:
                json.dump({"rules": rules}, f, ensure_ascii=False, indent=4)
    
    # Création des documents
    documents = [Document(page_content=rule) for rule in rules]
    
    # Initialisation des embeddings avec le modèle all-MiniLM-L6-v2
    print("Chargement du modèle d'embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Dossier de persistance locale
    persist_directory = "./chroma_db"
    
    # Vider le dossier ChromaDB existant (optionnel selon chroma_db version, bypass via suppression du dossier si besoin)
    # ou utiliser from_documents qui remplace / complète souvent selon les IDs.
    
    # Création et sauvegarde de la base vectorielle
    print("Création et persistance de la base vectorielle...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Appel de persist() pour forcer l'écriture sur le disque (utile pour certaines versions de Chroma)
    if hasattr(vectorstore, 'persist'):
        vectorstore.persist()
        
    print(f"Base vectorielle sauvegardée avec succès dans le dossier '{persist_directory}'.")

if __name__ == "__main__":
    init_database()
