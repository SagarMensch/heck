"""
Layer 4: Validation Engine + Knowledge Base Fuzzy Matching
==========================================================
Implements the 3-Layer Advanced Encyclopedia:
1. Trie Search for prefix matching.
2. Double Metaphone for phonetic OCR corrections.
3. Jaro-Winkler for typo scoring.
"""

import re
import logging
import jellyfish
from typing import Optional, List, Set, Dict

logger = logging.getLogger(__name__)

INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat", 
    "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh", 
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab", 
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh", 
    "uttarakhand", "west bengal", "delhi", "chandigarh", "puducherry"
}

INDIAN_CITIES = {
    "mumbai", "bombay", "delhi", "new delhi", "kolkata", "chennai", "bangalore", 
    "bengaluru", "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow", "chandigarh", 
    "bhopal", "patna", "nagpur", "indore", "thane", "kochi", "surat", "bhubaneswar"
}

LIC_PLANS = {
    "jeevan anand", "jeevan umang", "jeevan labh", "jeevan lakshya",
    "jeevan saathi", "jeevan shanti", "jeevan sugam", "jeevan tarun"
}

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.value = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.value = word

    def search(self, prefix: str) -> List[str]:
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Gather all words with this prefix
        results = []
        self._dfs(node, results)
        return results

    def _dfs(self, node: TrieNode, results: List[str]):
        if node.is_end_of_word:
            results.append(node.value)
        for child in node.children.values():
            self._dfs(child, results)

class ValidationKB:
    def __init__(self):
        self.city_trie = Trie()
        self.state_trie = Trie()
        
        for city in INDIAN_CITIES:
            self.city_trie.insert(city)
        for state in INDIAN_STATES:
            self.state_trie.insert(state)

    def fuzzy_match(self, ocr_text: str, category: str) -> tuple[str, float, str]:
        """
        Applies 3-layer fuzzy matching:
        1. Jaro-Winkler (Exact/Typo)
        2. Metaphone (Phonetic)
        """
        valid_list = set()
        if category == "city":
            valid_list = INDIAN_CITIES
        elif category == "state":
            valid_list = INDIAN_STATES
        elif category == "plan":
            valid_list = LIC_PLANS
        else:
            return ocr_text, 1.0, "Verified"

        ocr_text_clean = ocr_text.lower().strip()
        ocr_sound = jellyfish.metaphone(ocr_text_clean)
        
        best_match = ocr_text
        highest_score = 0.0

        for valid_word in valid_list:
            # 1. Jaro-Winkler Check
            jw_score = jellyfish.jaro_winkler_similarity(ocr_text_clean, valid_word)
            
            # 2. Metaphone Check
            valid_sound = jellyfish.metaphone(valid_word)
            
            if ocr_sound == valid_sound and jw_score > 0.75:
                # Phonetic match + reasonable JW score = Instant Win
                return valid_word.title(), 1.0, "Verified"
                
            if jw_score > highest_score:
                highest_score = jw_score
                best_match = valid_word

        if highest_score > 0.88:
            return best_match.title(), highest_score, "Verified"
        elif highest_score < 0.70:
            return ocr_text, highest_score, "Review Needed" # Hallucination flagged
        else:
            return best_match.title(), highest_score, "Review Needed"

# Instantiate Singleton
kb = ValidationKB()
