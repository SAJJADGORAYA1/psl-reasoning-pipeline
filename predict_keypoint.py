import json
from pathlib import Path
import time
import sys
import base64
import mimetypes
import requests
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
API_KEY = "AIzaSyBNsavVonNcYikKM0hTzPwtehVmFjPLJZo"  # Replace with actual key
VIDEO_MODEL = "gemini-1.5-flash-latest"  # Updated to more capable model
VOCAB_BANK = "psl_vocabulary_bank.json"
MAX_RETRIES = 5
INITIAL_DELAY = 1
MAX_DELAY = 30

# Strict structured prompt template with JSON output
STRUCTURED_PROMPT = (
    "Generate EXCLUSIVELY a JSON array describing this sign language video with these REQUIREMENTS:\n"
    "1. Break into 10 fixed 0.1-second segments from 0.0s to 1.0s\n"
    "2. For EACH segment include ONLY these EXACT keys in order:\n"
    "   - hand_shape (ONLY: 'open/relaxed', 'fist', 'index-pointed', 'flat-hand', 'claw')\n"
    "   - location (ONLY: 'at-sides', 'near-chin', 'near-face', 'chest-level', 'below-chest')\n"
    "   - movement (ONLY: 'stationary', 'upward', 'downward', 'circular', 'side-to-side')\n"
    "   - dominant_hand (ONLY: 'right', 'left', 'both', 'N/A')\n"
    "   - facial_expression (ONLY: 'neutral', 'smiling', 'raised-eyebrows')\n"
    "3. Use EXACT timing boundaries:\n"
    "   - Segment 1: 0.0s-0.1s\n"
    "   - Segment 2: 0.1s-0.2s\n"
    "   - Segment 3: 0.2s-0.3s\n"
    "   - Segment 4: 0.3s-0.4s\n"
    "   - Segment 5: 0.4s-0.5s\n"
    "   - Segment 6: 0.5s-0.6s\n"
    "   - Segment 7: 0.6s-0.7s\n"
    "   - Segment 8: 0.7s-0.8s\n"
    "   - Segment 9: 0.8s-0.9s\n"
    "   - Segment 10: 0.9s-1.0s\n\n"
    "OUTPUT FORMAT: A COMPACT JSON ARRAY (MINIMAL WHITESPACE) FOLLOWING THIS STRUCTURE:\n"
    """[{"segment":1,"start":0.0,"end":0.1,"hand_shape":"open/relaxed","location":"at-sides","movement":"stationary","dominant_hand":"N/A","facial_expression":"neutral"},...]"""
)

def describe_sign_from_video(video_path: Path) -> list:
    """Generates structured video description with retry logic"""
    print(f"📹 Analyzing video: {video_path.name}")
    
    if not API_KEY or "YOUR_GEMINI_API_KEY" in API_KEY:
        print(" ERROR: Missing valid Gemini API key")
        return None

    try:
        # Encode video
        mime_type, _ = mimetypes.guess_type(video_path) or "video/mp4"
        with open(video_path, "rb") as f:
            video_data = f.read()
            if len(video_data) > 20 * 1024 * 1024:  # 20MB limit
                print(f"⚠️ Video too large ({len(video_data)/1024/1024:.1f}MB), skipping")
                return None
            encoded_video = base64.b64encode(video_data).decode("utf-8")

        # Build payload
        payload = {
            "contents": [{
                "parts": [
                    {"text": STRUCTURED_PROMPT},
                    {"inline_data": {"mime_type": mime_type, "data": encoded_video}}
                ]
            }],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 4000,
                "response_mime_type": "application/json"
            }
        }

        # Retry logic with exponential backoff
        delay = INITIAL_DELAY
        for attempt in range(MAX_RETRIES):
            try:
                # API request
                response = requests.post(
                    url=f"https://generativelanguage.googleapis.com/v1beta/models/{VIDEO_MODEL}:generateContent?key={API_KEY}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=120
                )
                response.raise_for_status()
                
                # Extract and parse JSON response
                response_json = response.json()
                content = response_json["candidates"][0]["content"]["parts"][0]["text"]
                
                # Extract JSON from response
                json_match = re.search(r'\[[\s\S]*\]', content)
                if json_match:
                    json_str = json_match.group()
                    # Attempt to fix truncated JSON
                    if not json_str.strip().endswith(']'):
                        json_str = json_str.rstrip() + ']'
                    return json.loads(json_str)
                else:
                    print(f"No JSON found in response: {content[:200]}...")
                    return None
                
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if hasattr(e, 'response') else "Unknown"
                if status_code == 503 and attempt < MAX_RETRIES - 1:
                    # Exponential backoff with jitter
                    jitter = random.uniform(0.5, 1.5)
                    sleep_time = min(delay * jitter, MAX_DELAY)
                    print(f" Service Unavailable (503). Retry #{attempt+1} in {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    delay *= 2
                    continue
                else:
                    print(f" HTTP Error {status_code}: {str(e)}")
                    return None
            except json.JSONDecodeError as e:
                print(f" JSON Decode Error: {str(e)}")
                return None
            except Exception as e:
                print(f" Unexpected Error: {type(e).__name__} - {str(e)}")
                return None

        return None  # All retries failed
        
    except Exception as e:
        print(f" Processing Error: {str(e)[:200]}")
        return None

def categorical_to_vector(categories: dict) -> list:
    """Convert categorical features to numerical vector"""
    # Define category mappings
    mappings = {
        'hand_shape': ['open/relaxed', 'fist', 'index-pointed', 'flat-hand', 'claw'],
        'location': ['at-sides', 'near-chin', 'near-face', 'chest-level', 'below-chest'],
        'movement': ['stationary', 'upward', 'downward', 'circular', 'side-to-side'],
        'dominant_hand': ['right', 'left', 'both', 'N/A'],
        'facial_expression': ['neutral', 'smiling', 'raised-eyebrows']
    }
    
    vector = []
    for feature, options in mappings.items():
        # One-hot encoding for each feature
        encoding = [1 if categories[feature] == option else 0 for option in options]
        vector.extend(encoding)
    
    return vector

def segments_to_feature_vector(segments: list) -> list:
    """Convert structured segments to numerical feature vector"""
    feature_vector = []
    for seg in segments:
        # Convert each segment to numerical representation
        feature_vector.extend(categorical_to_vector(seg))
    return feature_vector

def predict_sign(test_video_path: Path, vocab_data: list):
    """Full prediction pipeline using structured feature vectors"""
    # Generate new description
    test_segments = describe_sign_from_video(test_video_path)
    if not test_segments or len(test_segments) != 10:
        print(f" Invalid segments generated: {len(test_segments) if test_segments else 0} segments")
        return None
    
    # Convert to feature vector
    test_vector = segments_to_feature_vector(test_segments)
    
    # Semantic comparison
    print(" Comparing with vocabulary bank...")
    
    # Prepare vocabulary vectors
    vocab_vectors = []
    valid_entries = []
    
    for entry in vocab_data:
        if isinstance(entry['description'], list) and len(entry['description']) == 10:
            vec = segments_to_feature_vector(entry['description'])
            vocab_vectors.append(vec)
            valid_entries.append(entry)
    
    if not vocab_vectors:
        print(" No valid vocabulary entries found")
        return None
    
    # Convert to numpy arrays for efficient computation
    vocab_matrix = np.array(vocab_vectors)
    test_array = np.array(test_vector).reshape(1, -1)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(test_array, vocab_matrix)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    return {
        "predicted_word": valid_entries[best_idx]['word'],
        "confidence_score": f"{best_score:.4f}",
        "input_video": test_video_path.name,
        "generated_segments": test_segments,
        "all_matches": [
            {
                "word": valid_entries[i]['word'],
                "similarity": f"{similarities[i]:.4f}"
            } for i in range(len(valid_entries))
        ]
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_from_video.py <path_to_your_video.mp4>")
        return
        
    test_video_path = Path(sys.argv[1])
    if not test_video_path.exists():
        print(f" Test video not found: {test_video_path}")
        return

    try:
        with open(VOCAB_BANK, "r") as f:
            vocab_data = json.load(f)
        print(f" Loaded vocabulary bank with {len(vocab_data)} entries.")
    except Exception as e:
        print(f" Error loading or parsing '{VOCAB_BANK}': {e}")
        return

    start_time = time.time()
    result = predict_sign(test_video_path, vocab_data)
    
    if result:
        print("\n" + "="*50)
        print(" PREDICTION RESULT")
        print(f"Predicted Word:     {result.get('predicted_word', 'N/A')}")
        print(f"Confidence Score:   {result.get('confidence_score', 'N/A')}")
        print(f"Input Video:        {result.get('input_video', 'N/A')}")
        
        # Show top 3 matches
        print("\n Top Matches:")
        matches = sorted(result['all_matches'], key=lambda x: float(x['similarity']), reverse=True)[:3]
        for i, match in enumerate(matches):
            print(f"{i+1}. {match['word']} (similarity: {match['similarity']})")
            
        print("="*50)
        print(f"\n Total time: {time.time() - start_time:.2f} seconds")

        output_file = f"prediction_{test_video_path.stem}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f" Saved full results to '{output_file}'")
    else:
        print("\n Prediction failed.")

if __name__ == "__main__":
    main()