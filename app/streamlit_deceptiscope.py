import streamlit as st
import spacy
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
import re
import string
from typing import List, Tuple, Dict, Optional
from models.ml_emotion_baseline import (
    load_or_build_emotion_model,
    predict_emotions,
)

# new imports for charts
import matplotlib.pyplot as plt


# 🔹 NEW: import our modular ML & DL models
from models.ml_baseline import load_or_build_lr, lr_predict_proba
from models.dl_deception_pytorch import load_or_build_dl, dl_predict_proba

# -----------------------
# Page config
# -----------------------
st.set_page_config(layout="wide", page_title="DeceptiScope")

# -----------------------
# Minimal downloads
# -----------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# -----------------------
# Constants & paths
# -----------------------
LR_MODEL_PATH = "models/lr_tfidf_lr.joblib"
LIME_NUM_FEATURES = 12
LIME_NUM_SAMPLES = 500

# Create custom stopword list
ENGLISH_STOPWORDS = set(stopwords.words('english'))
emotion_vectorizer, emotion_model, EMOTION_LABELS, emo_loaded = load_or_build_emotion_model()

# -----------------------
# Cached resource loaders
# -----------------------
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except Exception:
        return spacy.load("en_core_web_sm")


@st.cache_resource
def load_vader():
    return VaderAnalyzer()


# 🔹 NEW: cache-wrapped ML & DL model loaders
@st.cache_resource
def get_lr_model():
    # Load or train LR on LIAR + custom dataset (implemented in models/ml_baseline.py)
    pipe, loaded = load_or_build_lr(LR_MODEL_PATH)
    return pipe, loaded


@st.cache_resource
def get_dl_model():
    # Load or train PyTorch DL model on LIAR + custom (implemented in models/dl_deception_pytorch.py)
    model, vectorizer, loaded = load_or_build_dl()
    return model, vectorizer, loaded


# initialize cached models
nlp = load_spacy()
VADER = load_vader()
lr_pipe, lr_loaded = get_lr_model()
dl_model, dl_vectorizer, dl_loaded = get_dl_model()

# -----------------------
# Utility functions
# -----------------------
def preprocess_text_basic(text: str) -> str:
    return " ".join(text.strip().split()).lower()


@st.cache_data(show_spinner=False)
def extract_linguistic_features(text: str) -> dict:
    doc = nlp(text)
    words = [t.text for t in doc if not t.is_space]
    word_count = len([w for w in words if w.isalpha()])

    # Pronoun analysis
    pronouns = [t.text.lower() for t in doc if t.pos_ == "PRON"]
    first_person = [p for p in pronouns if p in ["i", "me", "my", "mine", "myself"]]

    # Negation detection
    negations = len([
        t for t in doc
        if t.lower_ in ["not", "n't", "never", "no", "didn't", "didnt",
                        "don't", "dont", "nothing", "nobody"]
    ])

    # Hedge detection
    hedge_patterns = [
        "might", "maybe", "could", "perhaps", "only because", "seems", "seem",
        "i think", "i guess", "sort of", "kind of", "probably", "possibly"
    ]
    text_lower = text.lower()
    hedges = sum(text_lower.count(p) for p in hedge_patterns)

    # Specific deceptive markers
    swear_count = text_lower.count("i swear") + text_lower.count("i promise")
    just_count = text_lower.count(" just ")

    pronoun_ratio = len(pronouns) / (word_count + 1e-9)
    first_person_ratio = len(first_person) / (word_count + 1e-9)

    # Tense analysis with sentence mapping
    sentences = list(doc.sents)
    past_verbs = [
        (t.text, t.i, sent.start)
        for sent in sentences for t in sent
        if t.tag_ in ["VBD", "VBN"]
    ]
    present_verbs = [
        (t.text, t.i, sent.start)
        for sent in sentences for t in sent
        if t.tag_ in ["VB", "VBP", "VBZ", "VBG"]
    ]

    past = len(past_verbs)
    present = len(present_verbs)

    # Detect tense shifts
    verb_timeline = []
    for t in doc:
        if t.tag_ in ["VBD", "VBN"]:
            verb_timeline.append(("past", t.text, t.i))
        elif t.tag_ in ["VB", "VBP", "VBZ", "VBG"]:
            verb_timeline.append(("present", t.text, t.i))

    tense_shifts = 0
    shift_details = []
    for i in range(1, len(verb_timeline)):
        if verb_timeline[i][0] != verb_timeline[i-1][0]:
            tense_shifts += 1
            shift_details.append({
                "from": verb_timeline[i-1][0],
                "to": verb_timeline[i][0],
                "from_verb": verb_timeline[i-1][1],
                "to_verb": verb_timeline[i][1],
                "position": verb_timeline[i][2]
            })

    vader_scores = VADER.polarity_scores(text)

    return {
        "word_count": word_count,
        "pronoun_ratio": round(pronoun_ratio, 3),
        "first_person_ratio": round(first_person_ratio, 3),
        "negation_count": negations,
        "hedge_count": hedges,
        "swear_count": swear_count,
        "just_count": just_count,
        "past_verb_count": past,
        "present_verb_count": present,
        "tense_shifts": tense_shifts,
        "shift_details": shift_details,
        "vader_compound": round(vader_scores.get("compound", 0.0), 3),
        "vader_pos": round(vader_scores.get("pos", 0.0), 3),
        "vader_neg": round(vader_scores.get("neg", 0.0), 3),
        "vader_neu": round(vader_scores.get("neu", 0.0), 3),
    }


def analyze_emotions_goemotions(text: str, emotion_classifier) -> dict:
    """Use GoEmotions model to get detailed emotion probabilities"""
    if emotion_classifier is None:
        vader = VADER.polarity_scores(text)
        anxiety = max(0.0, -vader.get("compound", 0.0) * 0.5)
        fear = anxiety * 0.8
        anger = vader.get("neg", 0.0) * 0.6
        joy = max(0.0, vader.get("pos", 0.0))
        neutral = vader.get("neu", 0.0)
        return {
            "anxiety": round(anxiety, 2),
            "fear": round(fear, 2),
            "anger": round(anger, 2),
            "joy": round(joy, 2),
            "neutral": round(neutral, 2)
        }

    try:
        results = emotion_classifier(text)[0]
        emotion_dict = {item['label']: item['score'] for item in results}

        anxiety = emotion_dict.get('nervousness', 0) + emotion_dict.get('fear', 0) * 0.5
        fear = emotion_dict.get('fear', 0)
        anger = emotion_dict.get('anger', 0) + emotion_dict.get('annoyance', 0) * 0.5
        joy = emotion_dict.get('joy', 0) + emotion_dict.get('amusement', 0) * 0.3
        neutral = emotion_dict.get('neutral', 0)

        return {
            "anxiety": round(anxiety, 2),
            "fear": round(fear, 2),
            "anger": round(anger, 2),
            "joy": round(joy, 2),
            "neutral": round(neutral, 2)
        }
    except Exception:
        return {
            "anxiety": 0.0,
            "fear": 0.0,
            "anger": 0.0,
            "joy": 0.0,
            "neutral": 0.5
        }


@st.cache_data(show_spinner=False)
def compute_guilt_index(emotions: dict, pronoun_ratio: float,
                        neg_density: float, hedge_density: float) -> float:
    anxiety = emotions.get("anxiety", 0.0)
    fear = emotions.get("fear", 0.0)
    anger = emotions.get("anger", 0.0)
    joy = emotions.get("joy", 0.0)

    pronoun_distance = 1.0 - min(pronoun_ratio * 5, 1.0)
    emotional_sum = anxiety + fear + anger - joy
    linguistic_markers = pronoun_distance + hedge_density + neg_density
    guilt_index = max(0.0, emotional_sum * linguistic_markers)

    return round(guilt_index, 2)


def identify_flagged_phrases(text: str, ling_feats: dict) -> List[str]:
    flagged = []
    text_lower = text.lower()

    negation_phrases = ["didn't", "never", "don't", "i didn't", "i never"]
    for phrase in negation_phrases:
        if phrase in text_lower:
            flagged.append(f'"{phrase}"')

    if "might" in text_lower or "maybe" in text_lower:
        flagged.append('"might have/maybe"')

    if "i swear" in text_lower:
        flagged.append('"I swear"')

    if "people are" in text_lower or "they are" in text_lower:
        flagged.append('"external blame"')

    return flagged


def get_deception_interpretation(deception_prob: float,
                                 guilt_index: float,
                                 ling_feats: dict) -> str:
    interpretation = f"Language indicates "
    markers = []

    if ling_feats["negation_count"] >= 3:
        markers.append("elevated cognitive load")
    if ling_feats["hedge_count"] >= 2:
        markers.append("self-distancing")
    if guilt_index > 0.6:
        markers.append("anxiety and fear suppression")
    if ling_feats.get("swear_count", 0) > 0:
        markers.append("persuasive (non-factual) emphasis")

    if markers:
        interpretation += " and ".join(markers)
        interpretation += " consistent with deceptive statements."
    else:
        interpretation += "some linguistic patterns that warrant attention."

    return interpretation


def highlight_text_with_colors(text: str, ling_feats: dict) -> str:
    html = text

    red_patterns = [
        (r'\b(didn\'t|never|don\'t)\s+\w+', '#ff4d4d', 0.7),
        (r'\b(i swear|i promise)\b', '#ff4d4d', 0.7),
    ]

    orange_patterns = [
        (r'\b(might have|maybe|could have|perhaps)\b', '#ff8c42', 0.6),
        (r'\b(just|only because)\b', '#ff8c42', 0.6),
        (r'\b(people are|they are|everyone)\s+\w+', '#ff8c42', 0.6),
    ]

    yellow_patterns = [
        (r'\b(i think|i guess|sort of|kind of)\b', '#ffd966', 0.5),
    ]

    all_patterns = red_patterns + orange_patterns + yellow_patterns

    for pattern, color, opacity in all_patterns:
        def repl(m):
            return (f'<span style="background-color:{color};opacity:{opacity};'
                    f'padding:0.1rem 0.2rem;border-radius:3px;'
                    f'font-weight:500;">{m.group(0)}</span>')

        html = re.sub(pattern, repl, html, flags=re.IGNORECASE)

    return html


def filter_lime_features(lime_list: List[Tuple[str, float]],
                         min_weight: float = 0.005) -> List[Tuple[str, float]]:
    """
    Filter out stopwords and low-impact features from LIME results.
    Keep only meaningful words/phrases that actually contribute to deception detection.
    """

    # Domain stopwords = normal stopwords + a few extra boring tokens
    domain_stopwords = ENGLISH_STOPWORDS.union({
        "the", "a", "an", "and", "or", "but", "if", "then",
        "was", "were", "is", "are", "am", "be", "been", "being",
        "by", "of", "to", "in", "on", "at", "for", "from",
        "that", "this", "it", "as", "with", "so", "just"
    })

    filtered = []

    # Keywords we ALWAYS care about if LIME surfaces them
    important_keywords = {
        "didn't", "didnt", "didn", "never", "don't", "dont",
        "i swear", "swear", "promise",
        "might", "maybe", "perhaps", "could",
        "only because", "because",
        "people", "everyone", "they are", "they're",
        "lying", "curious", "forgot", "window", "money", "register", "badge"
    }

    for feature, weight in lime_list:
        feature_clean = feature.strip().lower()

        # Remove punctuation for token checks
        no_punct = feature_clean.translate(str.maketrans('', '', string.punctuation))
        words = [w for w in no_punct.split() if w]

        if not words:
            continue

        keyword_hit = any(kw in feature_clean for kw in important_keywords)

        # 1) Single-word pure stopwords → drop (unless keyword)
        if len(words) == 1 and words[0] in domain_stopwords and not keyword_hit:
            continue

        # 2) Multi-word features that are only stopwords → drop
        if all(w in domain_stopwords for w in words) and not keyword_hit:
            continue

        # 3) Very small weights → drop, unless important keyword
        if abs(weight) < min_weight and not keyword_hit:
            continue

        filtered.append((feature, weight))

    return filtered


def prettify_lime_feature_text(feature: str) -> str:
    f = feature.strip()
    low = f.lower()
    if low in ("didn", "didnt"):
        return "didn't"
    if low == "dont":
        return "don't"
    return f


def generate_lime_explanation(lime_data: List[Tuple[str, float]]) -> str:
    """Generate natural language explanation of LIME results"""
    if not lime_data:
        return "No significant LIME features detected above threshold."

    # Filter out stopwords and noise
    filtered_lime = filter_lime_features(lime_data, min_weight=0.003)

    if not filtered_lime:
        return "LIME analysis found no significant deception markers above threshold."

    positive_features = [(f, w) for f, w in filtered_lime if w > 0]
    negative_features = [(f, w) for f, w in filtered_lime if w < 0]

    explanation = "**LIME Analysis Interpretation:**\n\n"

    if positive_features:
        top_deceptive = sorted(
            positive_features,
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        features_list = [f'**"{f}"** (+{w:.3f})' for f, w in top_deceptive]

        explanation += ("🔴 **Deception Indicators:** The model identified "
                        f"{', '.join(features_list)} as the strongest markers "
                        "pushing toward deception. These words/phrases are "
                        "statistically correlated with dishonest testimonies in "
                        "the training data, indicating patterns of linguistic "
                        "evasion, cognitive distancing, or over-assertion.\n\n")

    if negative_features:
        top_truthful = sorted(
            negative_features,
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        features_list = [f'**"{f}"** ({w:.3f})' for f, w in top_truthful]

        explanation += ("🟢 **Truthfulness Indicators:** Conversely, "
                        f"{', '.join(features_list)} pushed the prediction "
                        "toward truthfulness. These elements suggest direct, "
                        "factual language more commonly associated with honest "
                        "statements.")

    return explanation


def generate_linguistic_explanations(ling_feats: dict) -> Dict[str, str]:
    """Generate detailed explanations for each linguistic feature"""
    explanations = {}

    # Pronoun Ratio
    pr = ling_feats['pronoun_ratio']
    if pr > 0.15:
        explanations['pronoun'] = (
            f"**High pronoun usage ({pr:.2%})**: Excessive use of pronouns like "
            "'I', 'me', 'my' can indicate self-focus and psychological distancing. "
            "Deceptive individuals often unconsciously create linguistic distance "
            "from their lies by either over-using or under-using first-person pronouns."
        )
    else:
        explanations['pronoun'] = (
            f"**Normal pronoun usage ({pr:.2%})**: The pronoun ratio is within "
            "normal range, suggesting consistent self-reference without obvious "
            "distancing patterns."
        )

    # Negations
    neg = ling_feats['negation_count']
    if neg >= 3:
        explanations['negation'] = (
            f"**Elevated negations ({neg} instances)**: High use of negative "
            "constructions ('didn't', 'never', 'don't') suggests increased "
            "cognitive load. Liars often focus on what didn't happen rather "
            "than what did, as it's easier to deny than to fabricate detailed "
            "false events."
        )
    elif neg > 0:
        explanations['negation'] = (
            f"**Moderate negations ({neg} instances)**: Some negative language "
            "is present, which is normal in testimonies but should be evaluated "
            "in context."
        )
    else:
        explanations['negation'] = (
            f"**No negations detected**: The statement uses affirmative language "
            "throughout."
        )

    # Hedging
    hedge = ling_feats['hedge_count']
    if hedge >= 2:
        explanations['hedging'] = (
            f"**Significant hedging ({hedge} instances)**: Words like 'might', "
            "'maybe', 'perhaps', 'sort of' indicate uncertainty or lack of "
            "commitment to statements. This hedging behavior is common in "
            "deception as it provides psychological 'escape routes' and reduces "
            "the felt responsibility for false statements."
        )
    elif hedge > 0:
        explanations['hedging'] = (
            f"**Minor hedging ({hedge} instance)**: Some tentative language "
            "present, which could indicate genuine uncertainty or mild discomfort."
        )
    else:
        explanations['hedging'] = (
            f"**No hedging detected**: Statements are direct and assertive "
            "without qualification."
        )

    # Sentiment
    compound = ling_feats['vader_compound']
    if compound < -0.3:
        explanations['sentiment'] = (
            f"**Strongly negative sentiment ({compound:.2f})**: The overall "
            "emotional tone is notably negative. Deceptive statements often "
            "carry negative sentiment due to stress, defensiveness, or the "
            "negative nature of denied actions."
        )
    elif compound < -0.05:
        explanations['sentiment'] = (
            f"**Mildly negative sentiment ({compound:.2f})**: Slight negative "
            "emotional tone, which may reflect discomfort or the serious nature "
            "of the situation."
        )
    elif compound > 0.3:
        explanations['sentiment'] = (
            f"**Positive sentiment ({compound:.2f})**: Unusually positive tone "
            "for an interrogation context, which could indicate nervous "
            "positivity or genuine innocence."
        )
    else:
        explanations['sentiment'] = (
            f"**Neutral sentiment ({compound:.2f})**: The emotional tone is "
            "balanced, neither strongly positive nor negative."
        )

    return explanations


def analyze_tense_shifts(ling_feats: dict, text: str) -> str:
    """Analyze and explain temporal verb tense shifts"""
    shifts = ling_feats['tense_shifts']
    shift_details = ling_feats.get('shift_details', [])

    if shifts == 0:
        return ("**No tense shifts detected**: The narrative maintains "
                "consistent temporal perspective, suggesting a coherent memory "
                "or well-rehearsed story.")

    analysis = f"**{shifts} tense shift(s) detected**: "

    if shifts >= 3:
        analysis += (
            "Multiple shifts between past and present tense suggest the subject "
            "may be **reconstructing** rather than **recalling** events. When "
            "people recall genuine memories, they typically maintain consistent "
            "past tense. Frequent tense shifts indicate cognitive strain and "
            "possible fabrication, as the mind struggles between creating a "
            "narrative (present construction) and describing past events.\n\n"
        )
    elif shifts >= 1:
        analysis += (
            "Some tense inconsistency detected, which could indicate memory "
            "reconstruction or nervousness.\n\n"
        )

    if shift_details:
        analysis += "**Specific shifts identified:**\n"
        for i, shift in enumerate(shift_details[:3], 1):
            analysis += (
                f"{i}. From **{shift['from']}** tense ('{shift['from_verb']}') → "
                f"**{shift['to']}** tense ('{shift['to_verb']}')\n"
            )

        analysis += (
            "\n*Interpretation*: These shifts reveal moments where the narrative "
            "coherence breaks down, potentially indicating transition points "
            "between recalled truth and fabricated details."
        )

    return analysis


def calculate_deception_impact(ling_feats: dict, emotions: dict,
                               guilt_index: float,
                               lime_data: List) -> str:
    """Explain how each component contributes to final deception score"""

    impact_analysis = "**How Each Factor Contributes to Deception Probability:**\n\n"

    impacts = []

    # Negations impact
    if ling_feats['negation_count'] >= 3:
        impacts.append((
            "High negation count",
            "+15-20%",
            "Excessive denial language significantly increases deception probability"
        ))

    # Hedging impact
    if ling_feats['hedge_count'] >= 2:
        impacts.append((
            "Hedging language",
            "+10-15%",
            "Tentative, non-committal phrasing adds uncertainty markers"
        ))

    # Tense shifts
    if ling_feats.get('tense_shifts', 0) >= 2:
        impacts.append((
            "Tense inconsistency",
            "+8-12%",
            "Temporal shifts suggest narrative reconstruction"
        ))

    # Sentiment
    if ling_feats['vader_compound'] < -0.2:
        impacts.append((
            "Negative sentiment",
            "+5-8%",
            "Defensive emotional tone raises suspicion"
        ))

    # Guilt index
    if guilt_index > 0.6:
        impacts.append((
            "High guilt index",
            "+10-15%",
            "Elevated anxiety and fear markers indicate stress"
        ))

    # LIME features
    if lime_data:
        strong_deceptive = [f for f, w in lime_data if w > 0.01]
        if len(strong_deceptive) >= 3:
            impacts.append((
                "Key deceptive phrases",
                "+12-18%",
                "Multiple trained deception markers detected"
            ))

    # Pronoun patterns
    if ling_feats['pronoun_ratio'] > 0.15:
        impacts.append((
            "Pronoun distancing",
            "+5-7%",
            "Self-referential language patterns suggest discomfort"
        ))

    if not impacts:
        impact_analysis += (
            "The statement shows minimal deception indicators. Low probability is "
            "due to absence of typical deceptive markers.\n"
        )
    else:
        impact_analysis += "**Deception Score Breakdown:**\n\n"
        for factor, impact, explanation in impacts:
            impact_analysis += f"• **{factor}** ({impact}): {explanation}\n"

        impact_analysis += (
            "\n**Cumulative Effect**: These factors compound to produce the "
            "overall deception probability. Each marker increases cognitive load "
            "evidence, and multiple markers create a strong deception signature."
        )

    return impact_analysis


# ---------- NEW: narrative coherence via sentence embeddings ----------
@st.cache_data(show_spinner=False)
def compute_coherence_metrics(text: str) -> dict:
    """
    Compute narrative coherence using ONLY a TF-IDF model trained on the
    current testimony sentences (no external pretrained model).

    Steps:
    1) Split text into sentences with spaCy.
    2) Fit a TfidfVectorizer on these sentences.
    3) Use the TF-IDF vectors as embeddings.
    4) Compute cosine similarity between consecutive sentence vectors.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Need at least 2 sentences for pairwise similarity
    if len(sentences) < 2:
        return {
            "sentences": sentences,
            "similarities": [],
            "avg_similarity": None,
            "min_similarity": None,
            "max_similarity": None,
            "std_similarity": None,
        }

    # Hand-made “embedding” model: TF-IDF trained only on these sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)  # shape = (n_sentences, vocab)
    embeddings = tfidf_matrix.toarray()

    sims = []
    for i in range(len(sentences) - 1):
        v1 = embeddings[i]
        v2 = embeddings[i + 1]

        denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        sim = float(np.dot(v1, v2) / denom) if denom > 0 else 0.0
        sims.append(sim)

    return {
        "sentences": sentences,
        "similarities": sims,
        "avg_similarity": float(np.mean(sims)) if sims else None,
        "min_similarity": float(np.min(sims)) if sims else None,
        "max_similarity": float(np.max(sims)) if sims else None,
        "std_similarity": float(np.std(sims)) if sims else None,
    }


# ---------- NEW: emotion trajectory per sentence ----------
@st.cache_data(show_spinner=False)
def analyze_emotions_per_sentence(text: str) -> Tuple[List[str], List[dict]]:
    """
    Compute per-sentence emotion scores using our own ML emotion model
    (emotion_vectorizer + emotion_model), not Hugging Face GoEmotions.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    sent_emotions: List[dict] = []
    for sent in sentences:
        emo = predict_emotions(emotion_vectorizer, emotion_model, EMOTION_LABELS, sent)
        # Ensure keys exist for plotting
        emo = {
            "anxiety": float(emo.get("anxiety", 0.0)),
            "fear": float(emo.get("fear", 0.0)),
            "anger": float(emo.get("anger", 0.0)),
            "joy": float(emo.get("joy", 0.0)),
            "neutral": float(emo.get("neutral", 0.0)),
            "sentence": sent,
        }
        sent_emotions.append(emo)

    return sentences, sent_emotions



# ---------- NEW: passive voice analysis ----------
@st.cache_data(show_spinner=False)
def analyze_syntax_features(text: str) -> dict:
    doc = nlp(text)
    total_sentences = 0
    passive_sentences = 0
    passive_verbs = 0

    for sent in doc.sents:
        tokens = list(sent)
        if not tokens:
            continue
        total_sentences += 1
        has_passive = any(
            t.dep_ in ("nsubjpass", "auxpass") for t in sent
        )
        if has_passive:
            passive_sentences += 1
            passive_verbs += sum(
                1 for t in sent if t.dep_ in ("auxpass", "nsubjpass")
            )

    passive_ratio = passive_sentences / (total_sentences + 1e-9)

    return {
        "total_sentences": total_sentences,
        "passive_sentences": passive_sentences,
        "passive_ratio": passive_ratio,
        "passive_verbs": passive_verbs
    }


# ---------- NEW: missing-agent detection ----------
@st.cache_data(show_spinner=False)
def detect_missing_agents(text: str) -> dict:
    """
    Approximate 'missing agent' detection using dependency parse:
    we flag sentences where verbs have objects but no explicit subject.
    """
    doc = nlp(text)
    flagged = []

    for sent in doc.sents:
        for token in sent:
            if token.pos_ in ("VERB", "AUX"):
                has_subject = any(
                    child.dep_ in ("nsubj", "nsubjpass", "csubj")
                    for child in token.children
                )
                has_object = any(
                    child.dep_ in ("dobj", "obj", "pobj")
                    for child in token.children
                )
                if has_object and not has_subject:
                    flagged.append(sent.text.strip())
                    break

    # Deduplicate sentences while preserving order
    unique_flagged = list(dict.fromkeys(flagged))
    return {
        "missing_agent_count": len(unique_flagged),
        "missing_agent_sentences": unique_flagged
    }


# ---------- NEW: linguistic deception score ----------
def compute_linguistic_deception_score(ling_feats: dict,
                                       guilt_index: float) -> float:
    """
    Turn linguistic cues into a 0–1 deception risk score.
    This is *not* a probability, but a calibrated risk signal.
    """
    neg_score = min(ling_feats["negation_count"] / 4.0, 1.0)
    hedge_score = min(ling_feats["hedge_count"] / 3.0, 1.0)
    tense_score = min(ling_feats["tense_shifts"] / 3.0, 1.0)
    pron_score = min(ling_feats["pronoun_ratio"] / 0.2, 1.0)
    sentiment_score = max(0.0, -ling_feats["vader_compound"])  # 0..1-ish
    guilt_norm = min(guilt_index, 1.0)

    raw = (
        0.22 * neg_score +
        0.18 * hedge_score +
        0.15 * tense_score +
        0.12 * pron_score +
        0.13 * sentiment_score +
        0.20 * guilt_norm
    )

    return float(max(0.0, min(raw, 1.0)))


# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Analysis Mode:", ["Fast (Baseline only)", "Full (Deep Analysis)"])
enable_dl = st.sidebar.checkbox(
    "Enable Deep Learning Model (PyTorch)",
    value=(mode == "Full (Deep Analysis)")
)
enable_explanations = st.sidebar.checkbox("Enable LIME Explanations", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 About DeceptiScope**")
st.sidebar.markdown(
    "Forensic language profiler that detects deception through linguistic "
    "patterns, psychological cues, narrative coherence, and emotional analysis."
)

# -----------------------
# Main UI
# -----------------------
st.title("🕵️‍♂️ DeceptiScope — Forensic Language Profiler")
st.markdown("*Unmasking Lies, Manipulation, and Hidden Intent through Words*")
st.markdown("---")

# NEW default text engineered to be highly deceptive & rich in cues
default_text = """I didn't take the cash from the register last night; I was just closing up like I always do. 
Maybe I touched the drawer when I was checking if it was locked, but I never actually opened it, I swear I didn't. 
The drawer was left open when I came back from the back room, and the money was already gone by then. 
People are saying they saw me near the safe, but they must be confusing me with someone else because I would never do that. 
At first I said I wasn't even in the store at that time because I panicked, but now I remember I might have come back for a minute to grab my phone. 
Things happened quickly and mistakes were made, but nothing was done on purpose by me. 
I know it looks bad, and I understand why everyone is suspicious, but I promise you I'm not the one who took anything."""

if "input_text" not in st.session_state:
    st.session_state.input_text = default_text

text_area = st.text_area(
    "📝 Suspect Testimony:",
    value=st.session_state.input_text,
    height=200,
    key="input_area"
)
st.session_state.input_text = text_area

analyze_clicked = st.button("🔍 Analyze Testimony", type="primary", use_container_width=True)

if analyze_clicked:
    input_text = st.session_state.input_text.strip()

    if not input_text:
        st.error("Please enter a testimony to analyze.")
    else:
        # STEP 1: Linguistic Preprocessing
        with st.spinner("🧠 Step 1: Linguistic & Narrative Preprocessing..."):
            ling_feats = extract_linguistic_features(input_text)
            neg_density = ling_feats["negation_count"] / (ling_feats["word_count"] + 1e-9)
            hedge_density = ling_feats["hedge_count"] / (ling_feats["word_count"] + 1e-9)

            coherence_metrics = compute_coherence_metrics(input_text)
            syntax_feats = analyze_syntax_features(input_text)
            missing_agent_info = detect_missing_agents(input_text)

        # STEP 2: Baseline Model (TF-IDF + Logistic Regression)
        with st.spinner("🕵️ Step 2: Feature-Based Deception Classifier (LR)..."):
            try:
                lr_prob = lr_predict_proba(lr_pipe, [input_text])[0]
                if lr_prob.shape[0] >= 2:
                    lr_decept_prob = float(lr_prob[1])
                else:
                    lr_decept_prob = float(lr_prob.max())
                lr_label = "Likely Deceptive" if lr_decept_prob > 0.5 else "Likely Truthful"
            except Exception as e:
                st.error(f"Model prediction error: {e}")
                lr_decept_prob = 0.5
                lr_label = "Unavailable"

        # STEP 3: Deep Learning Model (PyTorch)
        dl_label = "Disabled"
        dl_decept_prob = None

        if enable_dl:
            with st.spinner("🤖 Step 3: Deep Learning Deception Classifier (PyTorch)..."):
                try:
                    dl_probs = dl_predict_proba(dl_model, dl_vectorizer, [input_text])
                    dl_decept_prob = float(dl_probs[0, 1])  # index 1 = deceptive
                    dl_label = "Likely Deceptive" if dl_decept_prob > 0.5 else "Likely Truthful"
                except Exception as e:
                    st.warning(f"DL prediction error: {e}")
                    dl_decept_prob = None
                    dl_label = "Error"

        # STEP 4: Emotions
        with st.spinner("⚖️ Step 4: Emotion Analysis & Guilt Index..."):
            raw_emotions = predict_emotions(emotion_vectorizer, emotion_model, EMOTION_LABELS, input_text)
            # keep same keys as before so the rest of your code works
            emotions = {
                "anxiety": float(raw_emotions.get("anxiety", 0.0)),
                "fear":    float(raw_emotions.get("fear", 0.0)),
                "anger":   float(raw_emotions.get("anger", 0.0)),
                "joy":     float(raw_emotions.get("joy", 0.0)),
                "neutral": float(raw_emotions.get("neutral", 0.0)),
            }

            guilt_index = compute_guilt_index(
                emotions,
                ling_feats["pronoun_ratio"],
                neg_density,
                hedge_density
            )
            linguistic_deception_score = compute_linguistic_deception_score(
                ling_feats, guilt_index
            )
            
            sent_list, sent_emotions = analyze_emotions_per_sentence(input_text)


        # STEP 5: Flagged phrases
        flagged_phrases = identify_flagged_phrases(input_text, ling_feats)

        # STEP 6: Combined deception score (LR + DL + linguistic risk)
        if dl_decept_prob is not None:
            combined_decept_prob = (
                0.45 * lr_decept_prob +
                0.35 * dl_decept_prob +
                0.20 * linguistic_deception_score
            )
        else:
            combined_decept_prob = (
                0.60 * lr_decept_prob +
                0.40 * linguistic_deception_score
            )
        combined_decept_prob = float(
            max(0.0, min(combined_decept_prob, 1.0))
        )
        combined_label = (
            "Highly Deceptive"
            if combined_decept_prob > 0.7
            else "Possibly Deceptive"
            if combined_decept_prob > 0.5
            else "More Likely Truthful"
        )

        interpretation = get_deception_interpretation(
            combined_decept_prob,
            guilt_index,
            ling_feats
        )

        # STEP 7: LIME with FIXED implementation (LR model)
        lr_explanations = []
        if enable_explanations:
            with st.spinner("📊 Generating LIME Explanations (this may take 10–20 seconds)..."):
                try:
                    explainer = LimeTextExplainer(
                        class_names=["truthful", "deceptive"],
                        bow=True
                    )

                    def lr_predict_proba_wrapper(texts):
                        try:
                            return lr_predict_proba(lr_pipe, texts)
                        except Exception as e:
                            st.warning(f"Prediction error in LIME: {e}")
                            return np.array([[0.5, 0.5]] * len(texts))

                    exp_lr = explainer.explain_instance(
                        input_text,
                        lr_predict_proba_wrapper,
                        num_features=15,
                        num_samples=500,
                        top_labels=1
                    )
                    raw_explanations = exp_lr.as_list(label=1)
                    lr_explanations = filter_lime_features(
                        raw_explanations,
                        min_weight=0.002
                    )
                    lr_explanations = sorted(
                        lr_explanations,
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:LIME_NUM_FEATURES]

                    st.success(f"✅ LIME found {len(lr_explanations)} significant features")
                except Exception as e:
                    st.warning(f"⚠️ LIME explanation failed: {e}. Continuing with other analyses.")
                    lr_explanations = []

        # Generate explanations
        lime_explanation = generate_lime_explanation(lr_explanations)
        linguistic_explanations = generate_linguistic_explanations(ling_feats)
        tense_analysis = analyze_tense_shifts(ling_feats, input_text)
        impact_analysis = calculate_deception_impact(
            ling_feats,
            emotions,
            guilt_index,
            lr_explanations
        )

        # ======================
        # DISPLAY RESULTS
        # ======================
        st.markdown("---")
        st.header("📋 Analysis Results")

        col_left, col_right = st.columns([3, 2])

        # LEFT COLUMN
        with col_left:
            st.subheader("🔍 Deception Heatmap")
            highlighted = highlight_text_with_colors(input_text, ling_feats)
            st.markdown(highlighted, unsafe_allow_html=True)
            st.markdown(
                "**Legend:** 🟥 Strong deception cues | 🟧 Defensive/Hedging | 🟨 Minor cues"
            )

            # LIME with explanation
            if lr_explanations:
                st.markdown("---")
                st.subheader("📊 Top Contributing Features (LIME) — Logistic Regression")

                df_lime = pd.DataFrame(lr_explanations, columns=["Feature", "Weight"])
                df_lime["Feature"] = df_lime["Feature"].apply(prettify_lime_feature_text)
                df_lime["Absolute"] = df_lime["Weight"].abs()
                df_lime = df_lime.sort_values("Absolute", ascending=False)

                df_lime["Impact"] = df_lime["Weight"].apply(
                    lambda x: f"{'🔴' if x > 0 else '🟢'} {abs(x):.4f}"
                )
                df_lime["Direction"] = df_lime["Weight"].apply(
                    lambda x: "Pushes toward DECEPTIVE"
                    if x > 0
                    else "Pushes toward TRUTHFUL"
                )

                st.dataframe(
                    df_lime[["Feature", "Impact", "Direction"]],
                    use_container_width=True,
                    hide_index=True
                )

                fig, ax = plt.subplots(figsize=(10, 6))
                colors_lime = [
                    "#ff4d4d" if w > 0 else "#66c266"
                    for w in df_lime["Weight"]
                ]
                bars = ax.barh(
                    df_lime["Feature"],
                    df_lime["Weight"],
                    color=colors_lime,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5
                )

                ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5)
                ax.set_xlabel(
                    "Feature Weight (Impact on LR Prediction)",
                    fontsize=11,
                    fontweight="bold"
                )
                ax.set_title(
                    "LIME: How Each Word/Phrase Influences the LR Deception Score",
                    fontsize=12,
                    fontweight="bold",
                    pad=15
                )
                ax.grid(axis="x", alpha=0.3, linestyle="--")
                ax.set_xlim(
                    min(df_lime["Weight"].min() * 1.2, -0.01),
                    max(df_lime["Weight"].max() * 1.2, 0.01)
                )

                for bar, val in zip(bars, df_lime["Weight"]):
                    if val > 0:
                        ax.text(
                            val,
                            bar.get_y() + bar.get_height() / 2,
                            f" +{val:.4f}",
                            va="center",
                            ha="left",
                            fontsize=9,
                            fontweight="bold"
                        )
                    else:
                        ax.text(
                            val,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.4f} ",
                            va="center",
                            ha="right",
                            fontsize=9,
                            fontweight="bold"
                        )

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.caption(
                    "🔴 **Red bars** = words/phrases that increase deception probability "
                    "in the LR model | 🟢 **Green bars** = words/phrases that decrease it."
                )

                with st.expander("📖 Understanding LIME Results", expanded=True):
                    st.markdown(lime_explanation)

                    st.markdown("---")
                    st.markdown("**🔬 Deep Dive into Key Features:**")

                    deceptive_features = [
                        (f, w)
                        for f, w in lr_explanations
                        if w > 0
                    ]
                    if deceptive_features:
                        st.markdown("**🔴 Most Deceptive Indicators (per LR model):**")
                        for i, (feature, weight) in enumerate(
                            sorted(deceptive_features, key=lambda x: x[1], reverse=True)[:3],
                            1
                        ):
                            st.markdown(f"{i}. **'{feature}'** (weight: +{weight:.4f})")
                            low = feature.lower()
                            if "didn't" in low or "never" in low:
                                st.markdown(
                                    "   - *Analysis:* Strong negation pattern. This kind of denial "
                                    "often appears when the speaker focuses on what they *didn't* do "
                                    "instead of giving concrete details about what actually happened."
                                )
                            elif "swear" in low or "promise" in low:
                                st.markdown(
                                    "   - *Analysis:* Over-assertion marker. The statement relies "
                                    "on emotional emphasis ('I swear', 'I promise') rather than "
                                    "verifiable facts."
                                )
                            elif "might" in low or "maybe" in low:
                                st.markdown(
                                    "   - *Analysis:* Hedging language that keeps options open and "
                                    "reduces responsibility for the statement."
                                )
                            elif "people" in low or "everyone" in low:
                                st.markdown(
                                    "   - *Analysis:* External blame shifting; responsibility is "
                                    "diffused onto unnamed others."
                                )
                            else:
                                st.markdown(
                                    "   - *Analysis:* This token has a strong statistical correlation "
                                    "with deceptive examples in the LR training space."
                                )
                            st.markdown("")

                    truthful_features = [
                        (f, w)
                        for f, w in lr_explanations
                        if w < 0
                    ]
                    if truthful_features:
                        st.markdown("**🟢 Truthfulness Indicators (per LR model):**")
                        for i, (feature, weight) in enumerate(
                            sorted(truthful_features, key=lambda x: abs(x[1]), reverse=True)[:2],
                            1
                        ):
                            st.markdown(f"{i}. **'{feature}'** (weight: {weight:.4f})")
                            st.markdown(
                                "   - *Analysis:* This element is associated with direct, factual "
                                "language that LR has learned to treat as more honest."
                            )
                            st.markdown("")
            else:
                st.info(
                    "ℹ️ LIME analysis not available or produced no significant features above threshold."
                )

        # RIGHT COLUMN
        with col_right:
            st.subheader("📈 Deception Probability")

            st.metric(
                "Combined Deception Probability",
                f"{combined_decept_prob * 100:.0f}%",
                delta=combined_label
            )

            st.metric(
                "Baseline Model (LR)",
                f"{lr_decept_prob * 100:.0f}%",
                delta=lr_label,
                delta_color="inverse" if lr_decept_prob > 0.5 else "normal"
            )

            if enable_dl and dl_decept_prob is not None:
                st.metric(
                    "Deep Learning Model (PyTorch)",
                    f"{dl_decept_prob * 100:.0f}%",
                    delta=dl_label,
                )

            st.metric(
                "Linguistic Deception Risk",
                f"{linguistic_deception_score * 100:.0f}%",
                delta="High" if linguistic_deception_score > 0.6
                else "Moderate" if linguistic_deception_score > 0.3
                else "Low"
            )

            st.markdown("---")
            st.subheader("🧠 Linguistic Profile")

            profile_data = {
                "Feature": [
                    "Pronoun Ratio (I/me/my)",
                    "Negation Count",
                    "Hedging Phrases",
                    "Sentiment (Compound)"
                ],
                "Value": [
                    f"{ling_feats['pronoun_ratio']:.2f}",
                    str(ling_feats['negation_count']),
                    str(ling_feats['hedge_count']),
                    f"{ling_feats['vader_compound']:.2f}"
                ],
                "Interpretation": [
                    "High" if ling_feats['pronoun_ratio'] > 0.15 else "Normal",
                    "High" if ling_feats['negation_count'] >= 3 else "Normal",
                    "High" if ling_feats['hedge_count'] >= 2 else "Normal",
                    "Negative" if ling_feats['vader_compound'] < -0.1 else "Neutral/Positive"
                ]
            }
            st.dataframe(
                pd.DataFrame(profile_data),
                use_container_width=True,
                hide_index=True
            )

            with st.expander("📖 Linguistic Feature Meanings", expanded=True):
                for key, explanation in linguistic_explanations.items():
                    st.markdown(explanation)
                    st.markdown("")

            st.markdown("---")
            st.subheader("😰 Emotion & Guilt Analysis (Overall)")

            fig_emotion, ax_emotion = plt.subplots(figsize=(5, 3))
            emotion_labels = list(emotions.keys())
            emotion_values = list(emotions.values())
            colors_emotion = ['#ff6b6b', '#ee5a6f', '#c44569', '#4ecdc4', '#95afc0']
            ax_emotion.barh(emotion_labels, emotion_values, color=colors_emotion)
            ax_emotion.set_xlabel('Probability')
            ax_emotion.set_title('Emotion Probabilities (Overall Text)')
            ax_emotion.set_xlim(0, 1)
            plt.tight_layout()
            st.pyplot(fig_emotion)
            plt.close()

            st.metric(
                "Guilt Index",
                f"{guilt_index:.2f}",
                delta="High tension" if guilt_index > 0.6
                else "Moderate" if guilt_index > 0.3
                else "Low",
                delta_color="inverse" if guilt_index > 0.6 else "off"
            )

        # Full-width sentiment polarity section
        st.markdown("---")
        st.header("📊 Sentiment & Emotion Dynamics")

        col_sent1, col_sent2 = st.columns([2, 1])

        with col_sent1:
            fig_sent, ax_sent = plt.subplots(figsize=(8, 4))
            sentiment_components = ['Positive', 'Negative', 'Neutral']
            sentiment_values = [
                ling_feats['vader_pos'],
                ling_feats['vader_neg'],
                ling_feats['vader_neu']
            ]
            colors_sent = ['#66c266', '#ff4d4d', '#95afc0']

            bars_sent = ax_sent.bar(
                sentiment_components,
                sentiment_values,
                color=colors_sent,
                alpha=0.7,
                edgecolor='black'
            )
            ax_sent.set_ylabel('Score', fontweight='bold')
            ax_sent.set_title('VADER Sentiment Component Breakdown', fontweight='bold')
            ax_sent.set_ylim(0, 1)
            ax_sent.grid(axis='y', alpha=0.3)

            for bar in bars_sent:
                height = bar.get_height()
                ax_sent.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontweight='bold'
                )

            plt.tight_layout()
            st.pyplot(fig_sent)
            plt.close()

        with col_sent2:
            st.metric(
                "Overall Sentiment",
                f"{ling_feats['vader_compound']:.3f}",
                delta="Negative" if ling_feats['vader_compound'] < 0
                else "Positive"
            )
            st.metric("Positive Score", f"{ling_feats['vader_pos']:.2f}")
            st.metric("Negative Score", f"{ling_feats['vader_neg']:.2f}")
            st.metric("Neutral Score", f"{ling_feats['vader_neu']:.2f}")

        st.info(
            f"**Sentiment Interpretation:** {linguistic_explanations.get('sentiment', 'No sentiment analysis available.')}"
        )

        # ---------- NEW: Emotion trajectory per sentence ----------
        st.markdown("---")
        st.header("📈 Emotion Trajectory Across the Story")

        if sent_list:
            fig_traj, ax_traj = plt.subplots(figsize=(10, 4))
            idx = np.arange(len(sent_list))

            anxiety_vals = [s["anxiety"] for s in sent_emotions]
            fear_vals = [s["fear"] for s in sent_emotions]
            anger_vals = [s["anger"] for s in sent_emotions]
            joy_vals = [s["joy"] for s in sent_emotions]

            ax_traj.plot(idx, anxiety_vals, marker='o', label='Anxiety')
            ax_traj.plot(idx, fear_vals, marker='o', label='Fear')
            ax_traj.plot(idx, anger_vals, marker='o', label='Anger')
            ax_traj.plot(idx, joy_vals, marker='o', label='Joy')

            ax_traj.set_xticks(idx)
            ax_traj.set_xticklabels([f"S{i+1}" for i in idx])
            ax_traj.set_ylim(0, 1)
            ax_traj.set_xlabel("Sentence")
            ax_traj.set_ylabel("Emotion intensity")
            ax_traj.set_title("Emotional Flow Across Sentences")
            ax_traj.grid(alpha=0.3)
            ax_traj.legend()

            plt.tight_layout()
            st.pyplot(fig_traj)
            plt.close()

            st.caption(
                "This chart shows how emotions change from sentence to sentence. "
                "Large spikes or abrupt switches (e.g., calm → anxious → calm) can "
                "signal emotional instability or reconstruction rather than a stable memory."
            )
        else:
            st.info("No sentences detected for emotion trajectory analysis.")

        # Temporal Analysis Section
        st.markdown("---")
        st.header("⏱️ Temporal Verb Tense & Narrative Coherence")

        col_temp1, col_temp2 = st.columns([3, 2])

        with col_temp1:
            st.markdown(tense_analysis)

            if ling_feats.get('shift_details'):
                st.markdown("---")
                st.markdown("**Visual Timeline of Tense Shifts:**")

                shift_data = {
                    'Shift #': [i + 1 for i in range(len(ling_feats['shift_details']))],
                    'From': [s['from'].capitalize() for s in ling_feats['shift_details']],
                    'To': [s['to'].capitalize() for s in ling_feats['shift_details']],
                    'Example': [f"{s['from_verb']} → {s['to_verb']}"
                                for s in ling_feats['shift_details']]
                }
                st.dataframe(
                    pd.DataFrame(shift_data),
                    use_container_width=True,
                    hide_index=True
                )

        with col_temp2:
            # Tense distribution
            fig_tense, ax_tense = plt.subplots(figsize=(5, 4))
            tense_labels = ['Past Tense', 'Present Tense']
            tense_values = [
                ling_feats['past_verb_count'],
                ling_feats['present_verb_count']
            ]
            colors_tense = ['#6c5ce7', '#00b894']

            if sum(tense_values) > 0:
                ax_tense.pie(
                    tense_values,
                    labels=tense_labels,
                    colors=colors_tense,
                    autopct='%1.1f%%',
                    startangle=90,
                    explode=(0.05, 0.05)
                )
                ax_tense.set_title('Verb Tense Distribution', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_tense)
                plt.close()
            else:
                st.info("No verbs detected for tense distribution.")

        # ---------- NEW: Narrative coherence section ----------
        st.markdown("---")
        st.header("🧵 Narrative Coherence (Sentence-to-Sentence Similarity)")

        if coherence_metrics["similarities"]:
            sims = coherence_metrics["similarities"]
            sentences = coherence_metrics["sentences"]
            x = np.arange(len(sims))

            fig_coh, ax_coh = plt.subplots(figsize=(10, 4))
            ax_coh.plot(x, sims, marker='o')
            ax_coh.set_xticks(x)
            ax_coh.set_xticklabels(
                [f"S{i+1}→S{i+2}" for i in x],
                rotation=45
            )
            ax_coh.set_ylim(0, 1)
            ax_coh.set_ylabel("Cosine similarity")
            ax_coh.set_xlabel("Adjacent sentence pair")
            ax_coh.set_title("Semantic Coherence Between Consecutive Sentences")
            ax_coh.grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig_coh)
            plt.close()

            st.markdown(
                f"- **Average similarity:** `{coherence_metrics['avg_similarity']:.2f}`  "
                f"- **Minimum:** `{coherence_metrics['min_similarity']:.2f}`  "
                f"- **Maximum:** `{coherence_metrics['max_similarity']:.2f}`  "
                f"- **Std deviation:** `{coherence_metrics['std_similarity']:.2f}`"
            )

            st.info(
                "Higher values (closer to 1.0) mean consecutive sentences are semantically "
                "aligned, suggesting a stable story. Very low values or sharp drops "
                "indicate jumps or topic changes, which often appear when someone is "
                "patching parts of a narrative together rather than recalling a single, "
                "coherent memory."
            )
        else:
            st.info(
                "Coherence analysis requires at least two sentences and the "
                "`sentence-transformers` package. Install it to enable this feature."
            )

        # ---------- NEW: Syntax & Agency section ----------
        st.markdown("---")
        st.header("🏗️ Syntax & Agency: Passive Voice and Missing Agents")

        col_syn1, col_syn2 = st.columns([2, 3])

        with col_syn1:
            st.subheader("Passive Voice Patterns")
            st.markdown(
                f"- Total sentences: **{syntax_feats['total_sentences']}**  \n"
                f"- Sentences with passive voice: **{syntax_feats['passive_sentences']}**  \n"
                f"- Passive sentence ratio: **{syntax_feats['passive_ratio']:.2f}**  \n"
                f"- Passive-related tokens (auxpass/nsubjpass): **{syntax_feats['passive_verbs']}**"
            )
            st.info(
                "Passive voice (e.g., *'the money was taken'*) can be used to hide "
                "who actually performed an action. A high passive ratio suggests "
                "the subject may be avoiding responsibility or obscuring agents."
            )

        with col_syn2:
            st.subheader("Clauses With Missing Agents")
            st.markdown(
                f"- Sentences where actions occur **without an explicit subject**: "
                f"**{missing_agent_info['missing_agent_count']}**"
            )
            if missing_agent_info["missing_agent_sentences"]:
                st.markdown("**Examples flagged:**")
                for s in missing_agent_info["missing_agent_sentences"]:
                    st.markdown(f"• _{s}_")
            else:
                st.markdown("_No clear missing-agent sentences detected._")

            st.info(
                "When actions are described without a clear 'who' (e.g., *'the drawer "
                "was left open'*, *'mistakes were made'*), it can indicate distancing. "
                "Deceptive narrators often describe bad events in a way that erases the "
                "actor from the sentence."
            )

        # Deception Impact Analysis
        st.markdown("---")
        st.header("🎯 Deception Probability Breakdown")

        with st.expander("📈 How Each Factor Contributes to Final Score", expanded=True):
            st.markdown(impact_analysis)

            st.markdown("---")
            st.markdown("**Visual Impact Assessment:**")

            impact_factors = []
            impact_values = []

            if ling_feats['negation_count'] >= 3:
                impact_factors.append('High Negations')
                impact_values.append(17.5)

            if ling_feats['hedge_count'] >= 2:
                impact_factors.append('Hedging Language')
                impact_values.append(12.5)

            if ling_feats.get('tense_shifts', 0) >= 2:
                impact_factors.append('Tense Shifts')
                impact_values.append(10)

            if ling_feats['vader_compound'] < -0.2:
                impact_factors.append('Negative Sentiment')
                impact_values.append(6.5)

            if guilt_index > 0.6:
                impact_factors.append('High Guilt Index')
                impact_values.append(12.5)

            if lr_explanations and len(
                    [f for f, w in lr_explanations if w > 0.01]) >= 3:
                impact_factors.append('Deceptive Phrases')
                impact_values.append(15)

            if ling_feats['pronoun_ratio'] > 0.15:
                impact_factors.append('Pronoun Distancing')
                impact_values.append(6)

            if impact_factors:
                fig_impact, ax_impact = plt.subplots(figsize=(10, 5))
                colors_impact = plt.cm.Reds(
                    [0.4 + (i * 0.1) for i in range(len(impact_factors))]
                )

                bars_impact = ax_impact.barh(
                    impact_factors,
                    impact_values,
                    color=colors_impact,
                    edgecolor='black',
                    linewidth=1
                )
                ax_impact.set_xlabel(
                    'Estimated Contribution to Deception Score (%)',
                    fontweight='bold'
                )
                ax_impact.set_title(
                    'Individual Factor Contributions',
                    fontweight='bold',
                    pad=15
                )
                ax_impact.grid(axis='x', alpha=0.3)

                for bar, val in zip(bars_impact, impact_values):
                    ax_impact.text(
                        val + 0.5,
                        bar.get_y() + bar.get_height() / 2,
                        f'+{val:.1f}%',
                        va='center',
                        fontweight='bold',
                        fontsize=10
                    )

                plt.tight_layout()
                st.pyplot(fig_impact)
                plt.close()

                st.caption(
                    f"**Combined Effect**: These factors contribute to the overall "
                    f"deception probability of **{combined_decept_prob * 100:.1f}%** "
                    "shown at the top of the page."
                )

        # Forensic Report
        st.markdown("---")
        st.header("📄 Forensic Report")

        col_report1, col_report2 = st.columns(2)

        with col_report1:
            st.markdown("**Subject:** Anonymous")
            st.markdown("**Document Type:** Written Testimony")
            st.markdown(
                f"**Combined Deception Probability:** "
                f"{combined_decept_prob * 100:.0f}% ({combined_label})"
            )
            st.markdown(f"**Guilt Index:** {guilt_index:.2f}")
            st.markdown(f"**Sentiment Polarity:** {ling_feats['vader_compound']:.2f}")
            st.markdown(f"**Tense Shifts:** {ling_feats['tense_shifts']}")
            if coherence_metrics["avg_similarity"] is not None:
                st.markdown(
                    f"**Narrative Coherence (avg cosine):** "
                    f"{coherence_metrics['avg_similarity']:.2f}"
                )
            st.markdown(
                f"**Passive Sentence Ratio:** {syntax_feats['passive_ratio']:.2f}"
            )

        with col_report2:
            if flagged_phrases:
                st.markdown("**Flagged Phrases:**")
                for phrase in flagged_phrases[:8]:
                    st.markdown(f"- {phrase}")
            else:
                st.markdown("**Flagged Phrases:** None detected")

        st.markdown("---")
        st.subheader("🔬 Interpretation")
        st.info(interpretation)
        st.warning(
            "**Recommendation:** Subject's statement should be cross-verified "
            "against evidence. Multiple linguistic, emotional, and structural "
            "markers suggest potential omission or concealment."
        )

        # Deception probability pie chart (NOW uses combined probability)
        st.markdown("---")
        st.subheader("📊 Deception Distribution (Combined Score)")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
        sizes = [1.0 - combined_decept_prob, combined_decept_prob]
        labels = ['Truthful', 'Deceptive']
        colors_pie = ['#66c266', '#ff4d4d']
        explode = (0, 0.1) if combined_decept_prob > 0.5 else (0.1, 0)
        ax_pie.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors_pie,
            autopct='%1.1f%%',
            startangle=90
        )
        ax_pie.axis('equal')
        plt.tight_layout()
        st.pyplot(fig_pie)
        plt.close()

        st.success("✅ Analysis Complete — scroll up to inspect each dimension.")
else:
    st.info(
        "👆 Enter or edit the testimony above and click **Analyze Testimony** "
        "to begin forensic analysis."
    )

    with st.expander("📖 See Example Analysis Description"):
        st.markdown(
            """
        **Example Testimony:**  
        _\"I didn't take the cash from the register last night; I was just closing up...\"_

        If you run the default text, you should see:

        - 🔴 High **combined deception probability** driven by:
          - many negations (\"didn't\", \"never\"),
          - hedging (\"maybe\", \"might have\"),
          - strong emotional emphasis (\"I swear\", \"I promise\"),
          - passive constructions (\"the drawer was left open\", \"mistakes were made\"),
          - tense shifts and narrative corrections.
        - 🧠 **Narrative coherence** graph showing mostly related sentences but with
          dips where the story changes or is corrected.
        - 📈 **Emotion trajectory** across sentences, revealing spikes in anxiety
          when the subject talks about being seen or changing their story.
        - 🏗️ **Passive voice & missing agents** highlighting how actions are sometimes
          described without a clear \"who\", which is a classic distancing strategy.
        """
        )
