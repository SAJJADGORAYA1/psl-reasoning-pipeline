import requests
import json
from pathlib import Path
import time
import mimetypes
import base64
import random  # For jitter in backoff

# Configuration
API_KEY = "AIzaSyBNsavVonNcYikKM0hTzPwtehVmFjPLJZo"
MODEL_NAME = "gemini-2.0-flash"
VIDEO_DIR = "Words"
OUTPUT_JSON = "psl_vocabulary_bank.json"
MAX_RETRIES = 5  # Max retry attempts per video
INITIAL_DELAY = 1  # Initial delay in seconds
MAX_DELAY = 30  # Maximum delay between retries

STRUCTURED_PROMPT = (
    "Generate EXCLUSIVELY a JSON array describing this sign language video with these REQUIREMENTS:\n"
    "1. Break into 10 fixed 0.1-second segments from 0.0s to 1.0s\n"
    "2. For EACH segment, derive values by observing these EXACT features:\n"
    "   - HAND SHAPE: Classify based on finger joint configurations (thumb/index/middle/ring/little) and palm\n"
    "   - LOCATION: Determine position relative to body landmarks (chin/face/chest) using wrist positions\n"
    "   - MOVEMENT: Track wrist trajectory and velocity\n"
    "   - DOMINANCE: Compare motion amplitude between hands\n"
    "   - FACIAL: Analyze eyebrows/eyes/mouth/jawline/cheeks\n"
    "3. For EACH segment include ONLY these EXACT keys in order:\n"
    "   - hand_shape (ONLY: 'open/relaxed', 'fist', 'index-pointed', 'flat-hand', 'claw')\n"
    "   - location (ONLY: 'at-sides', 'near-chin', 'near-face', 'chest-level', 'below-chest')\n"
    "   - movement (ONLY: 'stationary', 'upward', 'downward', 'circular', 'side-to-side')\n"
    "   - dominant_hand (ONLY: 'right', 'left', 'both', 'N/A')\n"
    "   - facial_expression (ONLY: 'neutral', 'smiling', 'raised-eyebrows')\n"
    "4. Use EXACT timing boundaries:\n"
    "   - Segment 1: 0.0s-0.1s | Segment 2: 0.1s-0.2s | ... | Segment 10: 0.9s-1.0s\n\n"
    "OBSERVATION PRIORITIES:\n"
    "• Hand shape: Classify by finger joint angles and thumb positioning\n"
    "• Location: Map wrist positions to body landmarks (nose/chin/chest)\n"
    "• Movement: Calculate wrist displacement vectors\n"
    "• Dominance: Compare range of motion between left/right wrists\n"
    "• Facial: Focus on eyebrow height, mouth curvature, and cheek tension\n\n"
    "OUTPUT FORMAT:\n"
    "[{\"segment\": 1, \"start\": 0.0, \"end\": 0.1, \"hand_shape\": ...}, ...]\n\n"
    "EXAMPLE OUTPUT:\n"
    """[
      {
        "segment": 1,
        "start": 0.0,
        "end": 0.1,
        "hand_shape": "open/relaxed",
        "location": "at-sides",
        "movement": "stationary",
        "dominant_hand": "N/A",
        "facial_expression": "neutral"
      },
      ... # 9 more segments
    ]"""
)

def describe_sign(video_path: Path) -> list:
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
                {"text": STRUCTURED_PROMPT},
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
            return json.loads(content)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503 and attempt < MAX_RETRIES - 1:
                # Exponential backoff with jitter
                jitter = random.uniform(0.5, 1.5)
                sleep_time = min(delay * jitter, MAX_DELAY)
                print(f"⚠️ Model overloaded. Retry #{attempt+1} in {sleep_time:.1f}s")
                time.sleep(sleep_time)
                delay *= 2  # Double the delay for next retry
                continue
            else:
                print(f"🚨 Permanent API Error: {str(e)}")
                if hasattr(e, 'response') and e.response.text:
                    print(f"Response: {e.response.text}")
                return None
        except Exception as e:
            print(f"🚨 Unexpected Error: {str(e)}")
            return None

    return None  # All retries failed

def generate_vocabulary():
    vocabulary = []
    video_dir = Path(VIDEO_DIR)

    if not video_dir.exists():
        print(f"❌ Directory not found: {video_dir}")
        return

    # Process video files
    video_files = list(video_dir.glob("*.*"))
    for i, video_path in enumerate(video_files):
        if video_path.suffix.lower() not in [".mp4", ".mov", ".avi"]:
            continue

        word = video_path.stem
        print(f"\n🔍 Processing ({i+1}/{len(video_files)}): {word}")

        description = describe_sign(video_path)
        if description:
            vocabulary.append({
                "word": word,
                "description": description,
                "video_file": video_path.name
            })
            print(f"✅ Success: Segments: {len(description)}")
        else:
            print(f"❌ Failed to process {word} after {MAX_RETRIES} attempts")

        time.sleep(2)  # Increased delay between videos

    # Save structured vocabulary bank
    if vocabulary:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(vocabulary, f, indent=2)
        print(f"\n🎉 Success! Saved {len(vocabulary)} sign descriptions to {OUTPUT_JSON}")
    else:
        print("\n❌ No valid sign descriptions generated")

if __name__ == "__main__":
    generate_vocabulary()