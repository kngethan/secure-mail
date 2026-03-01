import streamlit as st
import spacy
import os
import json
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from init_db import init_database, get_current_rules

# Compatibilité entre les versions de LangChain
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

# Configuration de la page Streamlit
st.set_page_config(page_title="Secure-Mail", layout="centered")

# Chargement du modèle spaCy pour le français
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("fr_core_news_sm")
    except OSError:
        st.error("⚠️ Le modèle spaCy 'fr_core_news_sm' n'est pas installé.")
        st.info("Veuillez lancer la commande : `python -m spacy download fr_core_news_sm`")
        st.stop()

nlp = load_spacy_model()

def anonymize_text(text: str) -> str:
    """Remplace les entités (PER, ORG, LOC) par [CONFIDENTIEL]."""
    # Si le texte manque de majuscules, on aide spaCy intelligemment
    if sum(1 for c in text if c.isupper()) < len(text) * 0.05:
        words = text.split()
        processed_words = []
        titles = ["mr", "mr.", "m.", "monsieur", "mme", "mme.", "madame", "mlle", "mademoiselle", "dr", "dr.", "docteur", "prof", "prof.", "professeur", "cher", "chère"]
        # Mots clés fréquents qui précèdent souvent une entreprise/organisation/lieu
        pre_org_loc_words = ["chez", "à", "de", "pour", "vers", "dans", "sur"]
        
        for i, word in enumerate(words):
            clean_word = word.strip(".,;:!?")
            # Met une majuscule si c'est le mot après un titre de civilité
            if i > 0 and words[i-1].lower().strip(".,;:!?") in titles:
                processed_words.append(word.capitalize())
            # Met une majuscule si c'est le mot après un mot clé d'entreprise/lieu
            elif i > 0 and words[i-1].lower().strip(".,;:!?") in pre_org_loc_words and len(clean_word) > 2:
                processed_words.append(word.capitalize())
            # Met une majuscule si c'est le dernier mot du texte (souvent la signature)
            elif i == len(words) - 1:
                processed_words.append(word.capitalize())
            else:
                processed_words.append(word)
                
        processed_text = " ".join(processed_words)
    else:
        processed_text = text
        
    doc = nlp(processed_text)
    anonymized = text # On garde le texte original pour ne pas altérer la casse
    
    # Itérer en ordre inverse pour ne pas fausser les indices de remplacement
    for ent in reversed(doc.ents):
        if ent.label_ in ["PER", "ORG", "LOC"]:
            anonymized = anonymized[:ent.start_char] + "[CONFIDENTIEL]" + anonymized[ent.end_char:]
    return anonymized

HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f).get("history", [])
            except json.JSONDecodeError:
                return []
    return []

def save_history(entry):
    history = load_history()
    history.insert(0, entry) # Ajouter au début
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"history": history}, f, ensure_ascii=False, indent=4)

@st.cache_resource
def init_rag_chain(tone="Formel"):
    """Initialise et met en cache la chaîne LangChain."""
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Chargement de la base vectorielle ChromaDB locale
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 3. LLM local via Ollama
    llm = OllamaLLM(model="mistral", temperature=0.0)
    
    # 4. Définition du Prompt pour le RAG
    prompt_template = f"""
    Tu es un correcteur orthographique et un reformulateur professionnel. Ton rôle est STRICTEMENT de réécrire le brouillon fourni pour le rendre **{tone}** et sans faute, TOUT EN GARDANT LE MÊME POINT DE VUE ET LA MÊME INTENTION.
    
    Règles de style de l'entreprise à appliquer :
    {{context}}
    
    Brouillon original à reformuler de manière {tone}:
    <brouillon>
    {{draft}}
    </brouillon>
    
    RÈGLES ABSOLUES DE REFORMULATION :
    1. TU ES L'EXPÉDITEUR DU BROUILLON. Ne réponds pas au message, reformule-le. Si l'auteur dit "je suis pas content", tu dois dire "je vous fais part de mon mécontentement".
    2. N'INVENTE AUCUNE INFORMATION. Si le brouillon demande un remboursement, demande le remboursement. N'invente pas des solutions comme "nous pouvons envoyer un nouveau modem" ou "venir chercher un modem". Tu reformules la plainte du client, tu n'es pas le service client de SFR.
    3. NE CONVERSE PAS. Ne donne aucune explication avant ou après ton e-mail.
    4. S'il y a la mention [CONFIDENTIEL], laisse-la exactement comme telle.
    5. C'est un simple e-mail, sois direct, court et concis.
    6. NE METS PAS de formule de salutation ou de signature générique ou inventée comme "[Votre Nom]", "[Votre Prénom]", ou "Cordialement" si ce n'est pas demandé dans les textes.
    
    E-MAIL FINAL (rédige uniquement le texte de l'e-mail) :
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    # 5. Formatage des documents de règles
    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)
    
    # 6. Création de la chaîne LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs, "draft": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- INTERFACE UTILISATEUR STREAMLIT ---

st.title("Secure-Mail")
st.subheader("Assistant de rédaction d'emails")

st.markdown("Rédigez vos e-mails en toute confidentialité.")

# Création des onglets
tab1, tab2, tab3 = st.tabs(["Nouveau Message", "Règles de réponse", "Historique"])

with tab1:
    tone_choice = st.radio(
        "Choisissez le ton de l'e-mail :",
        ["Formel", "Commercial", "Détendu"],
        horizontal=True
    )

    # Saisie du brouillon
    draft_input = st.text_area(
        "Entrez votre brouillon d'email :", 
        height=150,
        placeholder="Exemple : Bonjour Jean Dupont de chez Microsoft à Paris, j'espère que tu vas bien. Voici le devis demandé."
    )

    if st.button("Générer l'email"):
        if draft_input.strip():
            # 1. Phase d'anonymisation
            with st.spinner("Anonymisation des données sensibles en cours..."):
                anonymized_draft = anonymize_text(draft_input)
                
            st.markdown("### Brouillon anonymisé")
            st.info(anonymized_draft)
            
            # 2. Phase de génération RAG
            with st.spinner("Génération de l'email via Ollama et vos règles..."):
                try:
                    # Initialise la chaine spécifiquement avec le ton choisi
                    rag_chain = init_rag_chain(tone=tone_choice)
                    result = rag_chain.invoke(anonymized_draft)
                    st.markdown("### Email final généré")
                    st.success(result)
                    
                    # Sauvegarder dans l'historique
                    save_history({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "tone": tone_choice,
                        "draft": draft_input,
                        "anonymized": anonymized_draft,
                        "result": result
                    })
                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de la génération : {e}")
                    st.info("Vérifiez que le serveur Ollama est bien lancé (`ollama serve`) et que le modèle 'mistral' est téléchargé (`ollama run mistral`).")
        else:
            st.warning("Veuillez saisir un brouillon d'email avant de cliquer sur 'Générer'.")

with tab2:
    st.markdown("### Règles de réponse")
    st.markdown("Modifiez les règles qui dictent le comportement de l'assistant par défaut. Le modèle les respectera à chaque génération.")
    
    current_rules = get_current_rules()
    rules_text = st.text_area(
        "Modifiez les règles (une par ligne) :",
        value="\n".join(current_rules),
        height=200
    )
    
    if st.button("Enregistrer et recharger la base de données"):
        with st.spinner("Mise à jour de ChromaDB en cours..."):
            new_rules_list = [r.strip() for r in rules_text.split("\n") if r.strip()]
            init_database(custom_rules=new_rules_list)
            # Vider le cache de init_rag_chain pour forcer le rechargement du retriever
            init_rag_chain.clear()
            st.success("Règles mises à jour avec succès !")

with tab3:
    st.markdown("### Historique des e-mails générés")
    history_data = load_history()
    
    if not history_data:
        st.info("Aucun e-mail n'a encore été généré.")
    else:
        for idx, entry in enumerate(history_data):
            with st.expander(f"{entry['date']} - Ton : {entry['tone']}"):
                st.markdown("**Brouillon original :**")
                st.text(entry['draft'])
                st.markdown("**Email généré :**")
                st.success(entry['result'])

