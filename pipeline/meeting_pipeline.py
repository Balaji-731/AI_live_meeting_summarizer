# pipeline/meeting_pipeline.py
import os
import re
import logging
from datetime import datetime
from stt.stt_manager import get_full_transcript
from config.settings import AUDIO_SETTINGS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_meeting(audio_file):
    """Process a meeting recording and return transcript and summary."""
    try:
        logger.info("Starting meeting processing...")
        
        # Get transcript
        logger.info("Transcribing audio...")
        transcript = get_full_transcript(audio_file)
        
        if not transcript.strip():
            return "", "Error: No speech detected or empty transcript"
            
        # Generate summary
        logger.info("Generating summary...")
        summary = generate_summary(transcript)
        
        return transcript, summary
        
    except Exception as e:
        logger.error(f"Error processing meeting: {e}")
        return "", f"Error: {str(e)}"

def generate_summary(transcript, min_compression_ratio=0.3):
    """
    Generate a summary using a prompt-based approach with a language model.
    Uses either local BART model or falls back to a simpler extractive method.
    """
    try:
        # First try using a local model with a good prompt
        try:
            from transformers import pipeline, set_seed
            import torch
            
            logger.info("Using prompt-based summarization with BART...")
            
            # Set seed for reproducibility
            set_seed(42)
            
            # Initialize model
            model_name = "facebook/bart-large-cnn"
            summarizer = pipeline(
                "summarization", 
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Calculate target lengths
            input_length = len(transcript.split())
            max_length = min(max(100, input_length // 4), 512)
            min_length = max(30, max_length // 3)
            
            # Update the prompt and summarization code in generate_summary function
            prompt = f"""
            Extract key information from this meeting transcript. Focus on:
            1. Main topics discussed
            2. Decisions made
            3. Action items with responsible persons
            4. Important discussion points

            Transcript:
            {transcript[:3000]}  # Reduced context window for better focus

            Summary:
            """

            # Generate summary with prompt
            summary = summarizer(
                transcript[:3000],  # Pass the transcript directly, not the prompt
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                truncation=True
            )[0]['summary_text']
            # Clean up the summary
            summary = summary.strip()
            if not any(summary.endswith(p) for p in '.!?'):
                summary = re.sub(r'[.!?]*$', '.', summary) + '.'
                
            logger.info("Prompt-based summarization completed successfully")
            return summary
            
        except Exception as e:
            logger.warning(f"Prompt-based summarization failed: {str(e)}. Falling back to extractive method.")
            return _extractive_summary(transcript)
            
    except Exception as e:
        logger.error(f"Error in generate_summary: {e}")
        return _extractive_summary(transcript, aggressive=True)

def _calculate_similarity(text1, text2):
    """Calculate similarity ratio between two texts (0-1)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    if not text1 or not text2:
        return 0.0
        
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]

def _extractive_summary(transcript, aggressive=False):
    """
    Enhanced extractive summarization with NLP techniques.
    
    Args:
        transcript (str): The text to summarize
        aggressive (bool): If True, use more aggressive summarization
    """
    try:
        import re
        import numpy as np
        from collections import defaultdict
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer, PorterStemmer
        from nltk import pos_tag
        import string
        from heapq import nlargest

        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

        logger.info("Using enhanced extractive summarization...")
        
        # Initialize NLP tools
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        
        # Enhanced preprocessing with POS tagging
        def preprocess(text, lemmatize=True, stem=False):
            # Convert to lowercase and remove punctuation
            text = text.lower()
            words = tokenizer.tokenize(text)
            
            # POS tagging for better lemmatization
            if lemmatize:
                pos_tags = pos_tag(words)
                words = []
                for word, tag in pos_tags:
                    if tag.startswith('NN'):
                        pos = 'n'
                    elif tag.startswith('VB'):
                        pos = 'v'
                    else:
                        pos = 'a'
                    words.append(lemmatizer.lemmatize(word, pos))
            
            # Remove stopwords and short words
            words = [w for w in words if w not in stop_words and len(w) > 2]
            
            # Optional stemming
            if stem:
                words = [stemmer.stem(word) for word in words]
                
            return ' '.join(words)

        # Split into sentences using NLTK for better handling
        sentences = sent_tokenize(transcript)
        if not sentences:
            return ""

        # Calculate target summary length - more aggressive if needed
        if aggressive:
            target_length = min(max(2, int(len(sentences) * 0.15)), 5)  # 15% or max 5 sentences
        else:
            target_length = min(max(3, int(len(sentences) * 0.2)), 8)   # 20% or max 8 sentences
            
        logger.info(f"Target summary length: {target_length} sentences")

        # Preprocess sentences with different strategies
        preprocessed_sentences = [preprocess(sent, lemmatize=True) for sent in sentences]
        preprocessed_stemmed = [preprocess(sent, lemmatize=False, stem=True) for sent in sentences]
        
        # Create multiple vector representations
        try:
            # TF-IDF with n-grams
            tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_sentences)
            
            # Count vectorizer for frequency analysis
            count_vectorizer = CountVectorizer(ngram_range=(1, 2))
            count_matrix = count_vectorizer.fit_transform(preprocessed_stemmed)
            
            # Combine features
            from scipy.sparse import hstack
            sentence_vectors = hstack([tfidf_matrix, count_matrix])
            
        except Exception as e:
            logger.warning(f"Error in vectorization: {e}")
            # Fallback to simple extraction with more aggressive filtering
            return " ".join(sentences[:min(3, len(sentences))])

        # Calculate sentence importance using multiple features
        sentence_scores = []
        
        # Get document frequency for each term
        word_frequencies = np.asarray(sentence_vectors.sum(axis=0)).ravel()
        
        for i, (sentence, vector) in enumerate(zip(sentences, sentence_vectors)):
            # Base score from TF-IDF and count features
            tfidf_score = np.sum(vector.toarray())
            
            # Position-based scoring (U-shaped curve: higher at beginning and end)
            position = i / len(sentences)
            if position < 0.1 or position > 0.9:  # First and last 10%
                position_score = 1.6
            elif position < 0.2 or position > 0.8:  # Next 10% from start/end
                position_score = 1.3
            else:
                position_score = 0.8  # Penalize middle sentences
            
            # Length-based scoring (prefer medium-length sentences)
            words = word_tokenize(sentence)
            word_count = len(words)
            if 10 <= word_count <= 20:
                length_score = 1.3
            elif 5 <= word_count <= 30:
                length_score = 1.0
            else:
                length_score = 0.7
            
            # Title word scoring (sentences containing words from the first sentence)
            title_words = set(word_tokenize(sentences[0].lower()))
            title_word_score = 1.0
            if i > 0:  # Don't give extra points to the first sentence
                title_word_count = sum(1 for word in words if word.lower() in title_words)
                if title_word_count > 0:
                    title_word_score = 1.2
            
            # Combine scores with weights
            final_score = (
                0.4 * tfidf_score +
                0.3 * position_score +
                0.2 * length_score +
                0.1 * title_word_score
            )
            
            # Apply penalty for very short or very long sentences
            if word_count < 5 or word_count > 40:
                final_score *= 0.7
                
            sentence_scores.append((final_score, sentence, i))

        # Sort sentences by score and select top N
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Select sentences ensuring diversity
        selected_indices = []
        selected_vectors = []
        
        for score, sent, idx in sentence_scores:
            if len(selected_indices) >= target_length * 1.5:  # Oversample
                break
                
            # Skip very similar sentences
            if selected_vectors:
                current_vec = sentence_vectors[idx].toarray()
                similarities = [cosine_similarity(current_vec, v.reshape(1, -1))[0][0] 
                              for v in selected_vectors]
                if similarities and max(similarities) > 0.8:  # Skip if too similar
                    continue
                selected_vectors.append(current_vec)
            
            selected_indices.append(idx)
            
            if len(selected_indices) >= target_length * 1.5:  # Don't need too many
                break
        
        # Sort selected sentences back to original order and limit to target length
        selected_indices = sorted(selected_indices[:target_length])
        summary_sentences = [sentences[i] for i in selected_indices]
        
        # Post-processing
        summary = ' '.join(summary_sentences)
        
        # Clean up whitespace and ensure proper punctuation
        summary = re.sub(r'\s+', ' ', summary).strip()
        summary = re.sub(r'\s([.,!?])(?=\s|$)', r'\1', summary)  # Fix space before punctuation
        
        # Ensure the summary ends with proper punctuation
        if not any(summary.endswith(p) for p in '.!?'):
            summary = re.sub(r'[.!?]*$', '.', summary) + '.'
            
        # If summary is still too similar to original, try a different approach
        if _calculate_similarity(transcript, summary) > 0.7 and not aggressive:
            logger.info("Summary too similar to original, trying more aggressive approach")
            return _extractive_summary(transcript, aggressive=True)

        return summary

    except Exception as e:
        logger.error(f"Error in enhanced extractive summarization: {e}")
        # Fallback to simple extraction with more aggressive filtering
        try:
            sentences = sent_tokenize(transcript)
            # Take first, middle, and last sentences
            if len(sentences) > 3:
                mid = len(sentences) // 2
                selected = [sentences[0], sentences[mid], sentences[-1]]
                return ' '.join(selected)
            return ' '.join(sentences[:3]) if len(sentences) > 3 else ' '.join(sentences)
        except:
            return transcript[:500] + "..."