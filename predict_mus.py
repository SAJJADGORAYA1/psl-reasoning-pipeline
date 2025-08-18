import json
import time
from pathlib import Path
import argparse
import logging
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import mimetypes
import base64
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Constants
API_KEY = "AIzaSyBNsavVonNcYikKM0hTzPwtehVmFjPLJZo"
MODEL_NAME = "gemini-2.0-flash"
MAX_RETRIES = 5
INITIAL_DELAY = 1
MAX_DELAY = 30

DET_FIELDS = ['hands_used', 'primary_hand', 'two_hand_relation', 
              'contact_presence', 'body_region_primary', 'path_family']
DET_WEIGHTS = {
    'hands_used': 0.9,
    'primary_hand': 0.7,
    'two_hand_relation': 0.8,
    'contact_presence': 0.9,
    'body_region_primary': 0.95,
    'path_family': 0.85
}
DET_THRESHOLD = 0.65  # Lowered threshold
EMBEDDING_THRESHOLD = 0.45  # Lowered threshold

# New prompt structure
TEST_PROMPT = """You are given a short video of a Pakistan Sign Language (PSL) sign.
IMPORTANT: The video shows the same action performed twice - provide ONE description that covers both repetitions.

Fill the JSON exactly as specified below. Return ONLY the JSON object. No markdown, no extra text.

1) Deterministic fields (enums only — pick from allowed values; if uncertain, choose closest):
- hands_used: one | both
- primary_hand: left | right | none (use "none" if hands_used = "both")
- two_hand_relation: same-action | support-contact | alternating | stacked (use "same-action" if hands_used = "one")
- contact_presence: none | hand-hand | hand-body
- body_region_primary: face-head | neck | chest | torso | waist | neutral-front
- path_family: none | push-pull | open-close | linear | arc-circle | twist

2) Embedding fields (strict motion language). Use only these motion-spec tokens:
- Directions: forward | back | up | down | left | right | inward | outward | toward-body | away-from-body
- Paths: straight | arc | circle | small-arc | big-arc | zigzag | twist | open-close | pulse
- Extent: small | medium | large
- Repetition: single | repeated
- Coordination: together | mirror | parallel | counter
- Stability: static | slight-drift
- Targets (body refs): forehead | eye | cheek | chin | mouth | ear | neck | shoulder | chest | stomach | waist | neutral-front

Write four short lines, exactly in this order, each ≤ 12 words:
- arm_motion: (upper arms/forearms)
- hand_motion: (wrist/hand path & orientation changes)
- finger_motion: (extension/opposition/aperture)
- movement_summary: (≤12 words summary of the overall motion)

Rules:
- Use telegraphic phrases from the tokens above; join with commas.
- If still: write "static".
- If both hands same: start with "both:"; else use "left:" and "right:".

3) Reasoning-friendly fields:
- overall_description: 2–3 sentences in plain language; may use world-knowledge analogies.

REQUIRED JSON FORMAT:
{
  "hands_used": "one_or_both",
  "primary_hand": "left_or_right_or_none",
  "two_hand_relation": "same-action_or_support-contact_or_alternating_or_stacked",
  "contact_presence": "none_or_hand-hand_or_hand-body",
  "body_region_primary": "face-head_or_neck_or_chest_or_torso_or_waist_or_neutral-front",
  "path_family": "none_or_push-pull_or_open-close_or_linear_or_arc-circle_or_twist",
  "arm_motion": "motion_description_line",
  "hand_motion": "motion_description_line",
  "finger_motion": "motion_description_line",
  "movement_summary": "≤12_words_summary",
  "overall_description": "2-3_sentences_describing_the_visible_sign_only"
}"""

def describe_sign(video_path: Path) -> dict:
    """Send video to Gemini API with retry logic and exponential backoff"""
    # Get MIME type and encode video
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type:
        mime_type = "video/mp4"
    
    with open(video_path, "rb") as f:
        encoded_video = base64.b64encode(f.read()).decode("utf-8")

    # API configuration
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}
    
    # Generation config
    generation_config = {
        "temperature": 0,
        "max_output_tokens": 2000,
        "response_mime_type": "application/json"
    }

    payload = {
        "contents": [{
            "parts": [
                {"text": TEST_PROMPT},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": encoded_video
                    }
                }
            ]
        }],
        "generationConfig": generation_config
    }

    # Retry logic with exponential backoff
    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, headers=headers, params=params)
            response.raise_for_status()
            
            # Extract and parse JSON response
            response_json = response.json()
            content = response_json["candidates"][0]["content"]["parts"][0]["text"]
            
            # Clean response - sometimes Gemini wraps in markdown code blocks
            cleaned_content = content.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_content)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503 and attempt < MAX_RETRIES - 1:
                # Exponential backoff with jitter
                jitter = random.uniform(0.5, 1.5)
                sleep_time = min(delay * jitter, MAX_DELAY)
                logger.info(f"Model overloaded. Retry #{attempt+1} in {sleep_time:.1f}s")
                time.sleep(sleep_time)
                delay *= 2  # Double the delay for next retry
                continue
            else:
                logger.error(f"Permanent API Error: {str(e)}")
                if hasattr(e, 'response') and e.response.text:
                    logger.error(f"Response: {e.response.text}")
                return None
        except Exception as e:
            logger.error(f"Unexpected Error: {str(e)}")
            return None

    return None  # All retries failed

def load_vocabulary(vocab_file: str) -> list:
    """Load vocabulary bank with validation"""
    try:
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        
        # Validate structure
        valid_entries = []
        for entry in vocab:
            if 'description' in entry and isinstance(entry['description'], dict):
                # Ensure required fields exist
                required_fields = [
                    "hands_used", "primary_hand", "two_hand_relation",
                    "contact_presence", "body_region_primary", "path_family",
                    "arm_motion", "hand_motion", "finger_motion",
                    "movement_summary", "overall_description"
                ]
                if all(field in entry['description'] for field in required_fields):
                    valid_entries.append(entry)
                else:
                    logger.warning(f"Invalid entry skipped: {entry.get('word', 'Unknown')}")
            else:
                logger.warning(f"Entry missing description: {entry.get('word', 'Unknown')}")
        
        logger.info(f"Loaded vocabulary with {len(valid_entries)} valid entries")
        return valid_entries
    except Exception as e:
        logger.error(f"Error loading vocabulary: {str(e)}")
        return []

def calculate_deterministic_score(input_desc: dict, vocab_desc: dict) -> float:
    """Calculate weighted score for deterministic fields"""
    total_score = 0.0
    max_possible = 0.0
    
    for field in DET_FIELDS:
        weight = DET_WEIGHTS.get(field, 0.8)
        max_possible += weight
        
        input_val = input_desc.get(field, "").lower()
        vocab_val = vocab_desc.get(field, "").lower()
        
        # Special handling for primary_hand when hands_used is 'both'
        if field == 'primary_hand' and input_desc.get('hands_used') == 'both':
            if vocab_val == 'none' or vocab_val == input_val:
                total_score += weight
            continue
                
        # Exact match scoring
        if input_val == vocab_val:
            total_score += weight
    
    return total_score / max_possible if max_possible > 0 else 0

def create_motion_text(description: dict) -> str:
    """Create combined text for embedding from motion fields"""
    motion_fields = [
        description.get('arm_motion', ''),
        description.get('hand_motion', ''),
        description.get('finger_motion', ''),
        description.get('movement_summary', ''),
        description.get('overall_description', '')
    ]
    return " ".join(motion_fields)

def main(video_path: str, vocab_file: str = 'psl_vocabulary_bank.json'):
    start_time = time.time()
    
    # Load vocabulary
    vocab = load_vocabulary(vocab_file)
    if not vocab:
        logger.error("No valid vocabulary entries found")
        return
    
    logger.info(f"📹 Analyzing video: {Path(video_path).name}")
    
    # Generate description for input video
    input_desc = describe_sign(Path(video_path))
    if not input_desc:
        logger.error("Failed to generate description for input video")
        return
    
    # Create input motion text
    input_motion_text = create_motion_text(input_desc)
    
    # Load embedding model
    logger.info("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    input_embedding = model.encode([input_motion_text])[0].reshape(1, -1)
    
    # Stage 1: Deterministic field matching
    det_candidates = []
    for entry in vocab:
        score = calculate_deterministic_score(input_desc, entry['description'])
        if score >= DET_THRESHOLD:
            det_candidates.append({
                'entry': entry,
                'det_score': score
            })
    
    logger.info(f"Stage 1: {len(det_candidates)} candidates above deterministic threshold")
    
    # Stage 2: Embedding similarity matching
    emb_candidates = []
    if det_candidates:
        # Precompute motion texts for candidates
        candidate_texts = [create_motion_text(c['entry']['description']) for c in det_candidates]
        candidate_embeddings = model.encode(candidate_texts)
        
        # Compute similarities
        similarities = cosine_similarity(input_embedding, candidate_embeddings)[0]
        
        for i, sim in enumerate(similarities):
            if sim >= EMBEDDING_THRESHOLD:
                emb_candidates.append({
                    'entry': det_candidates[i]['entry'],
                    'det_score': det_candidates[i]['det_score'],
                    'sim_score': sim
                })
    
    logger.info(f"Stage 2: {len(emb_candidates)} candidates above embedding threshold")
    
    # Sort candidates by combined score
    if emb_candidates:
        emb_candidates.sort(key=lambda x: (x['sim_score'] * 0.7) + (x['det_score'] * 0.3), reverse=True)
    
    # Prepare results
    predicted_word = ""
    reasoning = "No candidates above similarity thresholds"
    if emb_candidates:
        top_candidate = emb_candidates[0]['entry']
        predicted_word = top_candidate['word']
        
        # Build detailed reasoning
        reasoning_lines = [
            f"Best match: '{predicted_word}'",
            f"Deterministic score: {emb_candidates[0]['det_score']:.2f}",
            f"Similarity score: {emb_candidates[0]['sim_score']:.2f}",
            "Key matches:"
        ]
        
        # Add key matching fields
        for field in DET_FIELDS:
            input_val = input_desc.get(field, '')
            vocab_val = top_candidate['description'].get(field, '')
            if input_val.lower() == vocab_val.lower():
                reasoning_lines.append(f"- {field}: {input_val}")
        
        reasoning = "\n".join(reasoning_lines)
    
    # Print results
    print("\n" + "="*50)
    print(" PREDICTION RESULT")
    print(f"Predicted Word:     {predicted_word}")
    print(f"Reasoning:          {reasoning}")
    print(f"Input Video:        {Path(video_path).name}")
    print(f"Det Candidates:     {len(det_candidates)}")
    print(f"Emb Candidates:     {len(emb_candidates)}")
    print("="*50 + "\n")
    
    # Save full results
    result = {
        "input_video": Path(video_path).name,
        "input_description": input_desc,
        "predicted_word": predicted_word,
        "reasoning": reasoning,
        "det_candidates": [{
            "word": c['entry']['word'],
            "score": c['det_score']
        } for c in det_candidates],
        "emb_candidates": [{
            "word": c['entry']['word'],
            "det_score": c['det_score'],
            "sim_score": c['sim_score']
        } for c in emb_candidates]
    }
    
    output_file = f"prediction_{Path(video_path).stem}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved full results to '{output_file}'")
    
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Pakistan Sign Language word from a video.')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--vocab', default='psl_vocabulary_bank.json', help='Path to vocabulary bank JSON file')
    args = parser.parse_args()
    
    main(args.video_path, args.vocab)