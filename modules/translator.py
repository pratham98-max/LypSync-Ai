"""
Translation Module using Facebook M2M100
Translates English text to Indic languages (Hindi, Tamil, Telugu, etc.)
This model is open and does not require gated repo access.
"""

import os
import torch
from typing import List, Dict, Optional, Union
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class IndicTranslator:
    """
    M2M100-based translator for English to Indic languages.
    Supports many Indian languages without gating requirements.
    """
    
    # Language code mappings for M2M100
    SUPPORTED_LANGUAGES = {
        "hi": "hi",   # Hindi
        "ta": "ta",   # Tamil
        "te": "te",   # Telugu
        "bn": "bn",   # Bengali
        "mr": "mr",   # Marathi
        "gu": "gu",   # Gujarati
        "kn": "kn",   # Kannada
        "ml": "ml",   # Malayalam
        "pa": "pa",   # Punjabi
        "or": "or",   # Odia
        "as": "as",   # Assamese
        "ur": "ur",   # Urdu
        "ne": "ne",   # Nepali
        "si": "si",   # Sinhala (Sinhalese)
        "sd": "sd",   # Sindhi
    }
    
    LANGUAGE_NAMES = {
        "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali",
        "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
        "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ur": "Urdu",
        "ne": "Nepali", "si": "Sinhala", "sd": "Sindhi"
    }

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the M2M100 translator.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Open source model from Facebook
        self.model_name = "facebook/m2m100_418M"
        
        logger.info(f"Loading M2M100 model on {self.device}...")
        
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.device)
        
        self.model.eval()
        logger.info("M2M100 model loaded successfully.")
    
    def translate(
        self, 
        text: Union[str, List[str]], 
        target_lang: str,
        source_lang: str = "en"
    ) -> Union[str, List[str]]:
        """
        Translate English text to target Indic language.
        
        Args:
            text: Input text or list of texts in English
            target_lang: Target language code (e.g., 'hi' for Hindi)
            source_lang: Source language code (default: 'en')
            
        Returns:
            Translated text or list of translated texts
        """
        if target_lang not in self.SUPPORTED_LANGUAGES:
            # Fallback for codes not in our mapping if they exist in M2M100
            target_lang_code = target_lang
        else:
            target_lang_code = self.SUPPORTED_LANGUAGES[target_lang]
        
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        
        logger.info(f"Translating {len(texts)} segment(s) to {self.LANGUAGE_NAMES.get(target_lang, target_lang)}")
        
        # Set source language
        self.tokenizer.src_lang = source_lang
        
        translated_texts = []
        
        # Process in batches or individually
        for t in texts:
            encoded_input = self.tokenizer(t, return_tensors="pt").to(self.device)
            
            # Generate translation
            generated_tokens = self.model.generate(
                **encoded_input, 
                forced_bos_token_id=self.tokenizer.get_lang_id(target_lang_code)
            )
            
            translated = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            translated_texts.append(translated)
        
        logger.info(f"Translation complete.")
        
        return translated_texts[0] if single_input else translated_texts
    
    def translate_segments(
        self, 
        segments: List[Dict], 
        target_lang: str
    ) -> List[Dict]:
        """
        Translate a list of segments (with timestamps) to target language.
        """
        if not segments:
            return []
            
        texts = [seg["text"] for seg in segments]
        translated_texts = self.translate(texts, target_lang)
        
        translated_segments = []
        for seg, trans_text in zip(segments, translated_texts):
            translated_segments.append({
                "text": trans_text,
                "original_text": seg["text"],
                "start": seg["start"],
                "end": seg["end"]
            })
        
        return translated_segments
    
    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """Get dictionary of supported language codes and names."""
        return cls.LANGUAGE_NAMES.copy()


if __name__ == "__main__":
    # Test translator
    translator = IndicTranslator()
    
    test_text = "Hello, how are you today?"
    result = translator.translate(test_text, "hi")
    print(f"English: {test_text}")
    print(f"Hindi: {result}")
