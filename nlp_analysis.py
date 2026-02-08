"""
NLP Analysis Module for Audio Transcription Project
Includes: Sentiment Analysis, NER, Summarization, Toxicity Detection,
          Keyword Extraction, Readability Scores, Topic Classification
"""

import re
import math
import logging
from typing import Dict, Any, List, Tuple
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1. SENTIMENT ANALYSIS
# ============================================================
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment using TextBlob (lexicon-based) and 
    optionally transformers for deep learning approach.
    
    Returns: polarity (-1 to 1), subjectivity (0 to 1), label
    """
    try:
        from textblob import TextBlob
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
        subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
        
        # Determine label
        if polarity > 0.1:
            label = "Positive"
            emoji = "😊"
        elif polarity < -0.1:
            label = "Negative"
            emoji = "😔"
        else:
            label = "Neutral"
            emoji = "😐"
        
        return {
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "label": label,
            "emoji": emoji,
            "confidence": round(abs(polarity), 2)
        }
    except ImportError:
        logger.warning("TextBlob not installed. Install with: pip install textblob")
        return {"error": "TextBlob not installed", "label": "Unknown", "emoji": "❓"}


def analyze_sentiment_transformer(text: str) -> Dict[str, Any]:
    """
    Deep learning sentiment analysis using Hugging Face transformers.
    More accurate but slower than lexicon-based approach.
    """
    try:
        from transformers import pipeline
        
        # Use a lightweight sentiment model
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512
        )
        
        # Truncate text for model
        truncated_text = text[:512] if len(text) > 512 else text
        result = sentiment_pipeline(truncated_text)[0]
        
        label = result['label']
        score = result['score']
        
        return {
            "label": label,
            "confidence": round(score, 3),
            "emoji": "😊" if label == "POSITIVE" else "😔",
            "model": "DistilBERT"
        }
    except ImportError:
        logger.warning("Transformers not installed")
        return {"error": "Transformers not installed"}
    except Exception as e:
        logger.error(f"Transformer sentiment error: {e}")
        return {"error": str(e)}


# ============================================================
# 2. NAMED ENTITY RECOGNITION (NER)
# ============================================================
def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities using spaCy.
    Categories: PERSON, ORG, GPE (places), DATE, MONEY, etc.
    """
    try:
        import spacy
        
        # Try loading the model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")
        
        doc = nlp(text)
        
        entities = {}
        entity_details = []
        
        for ent in doc.ents:
            label = ent.label_
            if label not in entities:
                entities[label] = []
            if ent.text not in entities[label]:
                entities[label].append(ent.text)
            entity_details.append({
                "text": ent.text,
                "label": label,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Entity type descriptions
        entity_descriptions = {
            "PERSON": "People/Names",
            "ORG": "Organizations",
            "GPE": "Countries/Cities/States",
            "DATE": "Dates",
            "TIME": "Times",
            "MONEY": "Monetary values",
            "PERCENT": "Percentages",
            "PRODUCT": "Products",
            "EVENT": "Events",
            "WORK_OF_ART": "Titles of works",
            "LAW": "Laws/Documents",
            "LANGUAGE": "Languages",
            "FAC": "Facilities/Buildings",
            "NORP": "Nationalities/Groups",
            "LOC": "Locations",
            "CARDINAL": "Numbers",
            "ORDINAL": "Ordinal numbers"
        }
        
        return {
            "entities": entities,
            "entity_details": entity_details,
            "entity_count": len(entity_details),
            "descriptions": entity_descriptions
        }
    except ImportError:
        logger.warning("spaCy not installed. Install with: pip install spacy")
        return {"error": "spaCy not installed", "entities": {}}


# ============================================================
# 3. TEXT SUMMARIZATION
# ============================================================
def summarize_text_extractive(text: str, num_sentences: int = 3) -> Dict[str, Any]:
    """
    Extractive summarization using TF-IDF scoring.
    Selects most important sentences from the original text.
    """
    import re
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= num_sentences:
        return {
            "summary": text,
            "method": "extractive",
            "original_sentences": len(sentences),
            "summary_sentences": len(sentences),
            "compression_ratio": 1.0
        }
    
    # Calculate word frequencies (TF)
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    
    # Remove stop words from scoring
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'is', 'was', 'are', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where', 'why',
        'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too', 'very'
    }
    
    # Score sentences
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        words_in_sentence = re.findall(r'\b\w+\b', sentence.lower())
        score = sum(word_freq[w] for w in words_in_sentence if w not in stop_words)
        # Normalize by sentence length
        score = score / (len(words_in_sentence) + 1)
        sentence_scores.append((i, score, sentence))
    
    # Get top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    # Sort by original order
    top_sentences = sorted(top_sentences, key=lambda x: x[0])
    
    summary = ' '.join([s[2] for s in top_sentences])
    
    return {
        "summary": summary,
        "method": "extractive (TF-IDF)",
        "original_sentences": len(sentences),
        "summary_sentences": num_sentences,
        "compression_ratio": round(len(summary) / len(text), 2)
    }


def summarize_text_abstractive(text: str, max_length: int = 150) -> Dict[str, Any]:
    """
    Abstractive summarization using Hugging Face transformers.
    Generates new sentences that capture the meaning.
    """
    try:
        from transformers import pipeline
        
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            truncation=True
        )
        
        # BART has a max input of ~1024 tokens
        truncated_text = text[:1024] if len(text) > 1024 else text
        
        result = summarizer(
            truncated_text,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )[0]
        
        return {
            "summary": result['summary_text'],
            "method": "abstractive (BART)",
            "model": "facebook/bart-large-cnn",
            "compression_ratio": round(len(result['summary_text']) / len(text), 2)
        }
    except ImportError:
        logger.warning("Transformers not installed")
        return summarize_text_extractive(text)  # Fallback
    except Exception as e:
        logger.error(f"Abstractive summarization error: {e}")
        return summarize_text_extractive(text)  # Fallback


# ============================================================
# 4. TOXICITY DETECTION
# ============================================================
def detect_toxicity(text: str) -> Dict[str, Any]:
    """
    Detect toxic/inappropriate content using ML model.
    Categories: toxic, severe_toxic, obscene, threat, insult, identity_hate
    """
    try:
        from transformers import pipeline
        
        # Use toxicity detection model
        toxicity_pipeline = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            truncation=True,
            max_length=512
        )
        
        truncated_text = text[:512] if len(text) > 512 else text
        result = toxicity_pipeline(truncated_text)[0]
        
        is_toxic = result['label'] == 'toxic' and result['score'] > 0.5
        
        return {
            "is_toxic": is_toxic,
            "label": result['label'],
            "confidence": round(result['score'], 3),
            "model": "toxic-bert",
            "status": "🔴 Toxic Content Detected" if is_toxic else "🟢 Clean Content"
        }
    except ImportError:
        # Fallback to rule-based detection
        return detect_toxicity_rulebased(text)
    except Exception as e:
        logger.error(f"Toxicity detection error: {e}")
        return detect_toxicity_rulebased(text)


def detect_toxicity_rulebased(text: str) -> Dict[str, Any]:
    """
    Rule-based toxicity detection as fallback.
    Uses curated word lists for different categories.
    """
    text_lower = text.lower()
    
    # Expanded toxic word categories
    toxic_words = {
        "profanity": ["shit", "fuck", "fucking", "damn", "hell", "ass", "bitch", "bastard", "crap", "piss"],
        "slurs": ["idiot", "stupid", "dumb", "moron", "retard", "loser"],
        "threats": ["kill", "die", "murder", "destroy", "attack", "hurt"],
        "hate": ["hate", "racist", "sexist", "disgusting"]
    }
    
    found_toxic = {}
    total_toxic_count = 0
    
    for category, words in toxic_words.items():
        found = []
        for word in words:
            matches = re.findall(r'\b' + re.escape(word) + r'\b', text_lower)
            if matches:
                found.append(f"{word} ({len(matches)}x)")
                total_toxic_count += len(matches)
        if found:
            found_toxic[category] = found
    
    is_toxic = total_toxic_count > 0
    
    # Calculate toxicity score (0-1)
    word_count = len(text.split())
    toxicity_score = min(total_toxic_count / max(word_count, 1) * 10, 1.0)
    
    return {
        "is_toxic": is_toxic,
        "toxicity_score": round(toxicity_score, 3),
        "toxic_words_found": found_toxic,
        "total_toxic_count": total_toxic_count,
        "method": "rule-based",
        "status": "🔴 Toxic Content Detected" if is_toxic else "🟢 Clean Content"
    }


# ============================================================
# 5. KEYWORD EXTRACTION (TF-IDF)
# ============================================================
def extract_keywords_tfidf(text: str, top_n: int = 10) -> Dict[str, Any]:
    """
    Extract keywords using TF-IDF scoring.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Split into sentences for document-level TF-IDF
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) < 2:
            sentences = [text]  # Use whole text as single document
        
        # TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1
        )
        
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum TF-IDF scores across all sentences
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        
        # Get top keywords
        top_indices = tfidf_scores.argsort()[-top_n:][::-1]
        keywords = [(feature_names[i], round(tfidf_scores[i], 3)) for i in top_indices]
        
        return {
            "keywords": keywords,
            "method": "TF-IDF",
            "ngram_range": "1-2 (words & phrases)",
            "total_features": len(feature_names)
        }
    except ImportError:
        logger.warning("scikit-learn not installed")
        return extract_keywords_frequency(text, top_n)


def extract_keywords_frequency(text: str, top_n: int = 10) -> Dict[str, Any]:
    """
    Simple frequency-based keyword extraction (fallback).
    """
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'is', 'was', 'are', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'
    }
    
    words = re.findall(r'\b\w+\b', text.lower())
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    word_freq = Counter(words).most_common(top_n)
    
    return {
        "keywords": word_freq,
        "method": "frequency",
        "total_words": len(words)
    }


# ============================================================
# 6. READABILITY SCORES
# ============================================================
def calculate_readability(text: str) -> Dict[str, Any]:
    """
    Calculate various readability metrics.
    """
    # Count sentences, words, syllables
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)
    
    words = re.findall(r'\b\w+\b', text)
    word_count = max(len(words), 1)
    
    # Count syllables (approximation)
    def count_syllables(word):
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        # Handle silent e
        if word.endswith('e') and count > 1:
            count -= 1
        return max(count, 1)
    
    syllable_count = sum(count_syllables(w) for w in words)
    
    # Complex words (3+ syllables)
    complex_words = sum(1 for w in words if count_syllables(w) >= 3)
    
    # Average calculations
    avg_sentence_length = word_count / sentence_count
    avg_syllables_per_word = syllable_count / word_count
    
    # Flesch Reading Ease (0-100, higher = easier)
    flesch_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    flesch_ease = max(0, min(100, flesch_ease))
    
    # Flesch-Kincaid Grade Level
    fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    fk_grade = max(0, fk_grade)
    
    # Gunning Fog Index
    fog_index = 0.4 * (avg_sentence_length + 100 * (complex_words / word_count))
    
    # SMOG Index (requires 30+ sentences ideally)
    smog_index = 1.0430 * math.sqrt(complex_words * (30 / sentence_count)) + 3.1291
    
    # Interpret Flesch score
    if flesch_ease >= 90:
        level = "Very Easy (5th grade)"
    elif flesch_ease >= 80:
        level = "Easy (6th grade)"
    elif flesch_ease >= 70:
        level = "Fairly Easy (7th grade)"
    elif flesch_ease >= 60:
        level = "Standard (8th-9th grade)"
    elif flesch_ease >= 50:
        level = "Fairly Difficult (10th-12th grade)"
    elif flesch_ease >= 30:
        level = "Difficult (College)"
    else:
        level = "Very Difficult (Graduate)"
    
    return {
        "flesch_reading_ease": round(flesch_ease, 1),
        "flesch_kincaid_grade": round(fk_grade, 1),
        "gunning_fog_index": round(fog_index, 1),
        "smog_index": round(smog_index, 1),
        "reading_level": level,
        "statistics": {
            "sentences": sentence_count,
            "words": word_count,
            "syllables": syllable_count,
            "complex_words": complex_words,
            "avg_sentence_length": round(avg_sentence_length, 1),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2)
        }
    }


# ============================================================
# 7. POS TAGGING & ANALYSIS
# ============================================================
def analyze_pos_tags(text: str) -> Dict[str, Any]:
    """
    Part-of-Speech tagging and analysis using spaCy.
    """
    try:
        import spacy
        
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")
        
        doc = nlp(text)
        
        # Count POS tags
        pos_counts = Counter([token.pos_ for token in doc])
        
        # Get examples of each POS
        pos_examples = {}
        for token in doc:
            if token.pos_ not in pos_examples:
                pos_examples[token.pos_] = []
            if len(pos_examples[token.pos_]) < 5:
                pos_examples[token.pos_].append(token.text)
        
        # POS descriptions
        pos_descriptions = {
            "NOUN": "Nouns (things, people, places)",
            "VERB": "Verbs (actions)",
            "ADJ": "Adjectives (descriptors)",
            "ADV": "Adverbs (how/when/where)",
            "PRON": "Pronouns (I, you, they)",
            "DET": "Determiners (the, a, this)",
            "ADP": "Prepositions (in, on, at)",
            "CONJ": "Conjunctions (and, but, or)",
            "CCONJ": "Coordinating Conjunctions",
            "SCONJ": "Subordinating Conjunctions",
            "NUM": "Numbers",
            "PUNCT": "Punctuation",
            "INTJ": "Interjections (oh, wow)",
            "PROPN": "Proper Nouns (names)",
            "AUX": "Auxiliary Verbs (is, have)",
            "PART": "Particles (not, 's)",
            "SPACE": "Spaces",
            "SYM": "Symbols",
            "X": "Other"
        }
        
        return {
            "pos_counts": dict(pos_counts.most_common()),
            "pos_examples": pos_examples,
            "pos_descriptions": pos_descriptions,
            "total_tokens": len(doc)
        }
    except ImportError:
        return {"error": "spaCy not installed"}


# ============================================================
# 8. TOPIC CLASSIFICATION
# ============================================================
def classify_topic(text: str) -> Dict[str, Any]:
    """
    Zero-shot topic classification using transformers.
    """
    try:
        from transformers import pipeline
        
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            truncation=True
        )
        
        # Define candidate topics
        candidate_labels = [
            "Technology", "Business", "Politics", "Sports", "Entertainment",
            "Science", "Health", "Education", "Travel", "Food",
            "Music", "Art", "Religion", "History", "Nature"
        ]
        
        truncated_text = text[:512] if len(text) > 512 else text
        
        result = classifier(truncated_text, candidate_labels, multi_label=True)
        
        # Get top 5 topics
        topics = list(zip(result['labels'][:5], result['scores'][:5]))
        topics = [(label, round(score, 3)) for label, score in topics]
        
        return {
            "top_topics": topics,
            "primary_topic": result['labels'][0],
            "confidence": round(result['scores'][0], 3),
            "model": "BART-MNLI (zero-shot)"
        }
    except ImportError:
        return {"error": "Transformers not installed", "top_topics": []}
    except Exception as e:
        logger.error(f"Topic classification error: {e}")
        return {"error": str(e)}


# ============================================================
# COMPREHENSIVE NLP ANALYSIS
# ============================================================
def run_full_nlp_analysis(text: str, use_transformers: bool = False) -> Dict[str, Any]:
    """
    Run all NLP analyses on the given text.
    
    Args:
        text: Input text to analyze
        use_transformers: If True, use heavy transformer models (slower but more accurate)
    
    Returns:
        Dict with all analysis results
    """
    logger.info("Starting comprehensive NLP analysis...")
    
    results = {}
    
    # 1. Sentiment Analysis
    logger.info("Analyzing sentiment...")
    results["sentiment"] = analyze_sentiment(text)
    if use_transformers:
        results["sentiment_deep"] = analyze_sentiment_transformer(text)
    
    # 2. Named Entity Recognition
    logger.info("Extracting named entities...")
    results["entities"] = extract_named_entities(text)
    
    # 3. Text Summarization
    logger.info("Generating summary...")
    results["summary"] = summarize_text_extractive(text)
    if use_transformers:
        results["summary_abstractive"] = summarize_text_abstractive(text)
    
    # 4. Toxicity Detection
    logger.info("Detecting toxicity...")
    if use_transformers:
        results["toxicity"] = detect_toxicity(text)
    else:
        results["toxicity"] = detect_toxicity_rulebased(text)
    
    # 5. Keyword Extraction
    logger.info("Extracting keywords...")
    results["keywords"] = extract_keywords_tfidf(text)
    
    # 6. Readability Scores
    logger.info("Calculating readability...")
    results["readability"] = calculate_readability(text)
    
    # 7. POS Tagging
    logger.info("Analyzing parts of speech...")
    results["pos_analysis"] = analyze_pos_tags(text)
    
    # 8. Topic Classification (only with transformers)
    if use_transformers:
        logger.info("Classifying topics...")
        results["topics"] = classify_topic(text)
    
    logger.info("NLP analysis complete!")
    return results
