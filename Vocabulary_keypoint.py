import requests
import json
from pathlib import Path
import time
import mimetypes
import base64
import random
import re
import logging

# Configuration
API_KEY = "AIzaSyBNsavVonNcYikKM0hTzPwtehVmFjPLJZo"
MODEL_NAME = "gemini-1.5-flash-latest"  # Updated to more capable model
VIDEO_DIR = "Words"
OUTPUT_JSON = "psl_vocabulary_bank.json"
MAX_RETRIES = 5
INITIAL_DELAY = 1
MAX_DELAY = 30

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STRUCTURED_PROMPT = (
    "Generate EXCLUSIVELY a JSON array describing this sign language video with these REQUIREMENTS:\n"
    "1. Break into 10 fixed 0.1-second segments from 0.0s to 1.0s\n"
    "2. For EACH segment include:\n"
    "   A. The original classification keys (IN ORDER):\n"
    "      - hand_shape (ONLY: 'open/relaxed', 'fist', 'index-pointed', 'flat-hand', 'claw')\n"
    "      - location (ONLY: 'at-sides', 'near-chin', 'near-face', 'chest-level', 'below-chest')\n"
    "      - movement (ONLY: 'stationary', 'upward', 'downward', 'circular', 'side-to-side')\n"
    "      - dominant_hand (ONLY: 'right', 'left', 'both', 'N/A')\n"
    "      - facial_expression (ONLY: 'neutral', 'smiling', 'raised-eyebrows')\n"
    "   B. Detailed observations of these EXACT features:\n"
    "      - head_neck: {nose_position: [x,y,z], eye_aperture: float, ear_visibility: bool}\n"
    "      - upper_body: {shoulder_angle: float, elbow_flexion: [left,right], wrist_position: [left,right]}\n"
    "      - hand_joints (PER HAND):\n"
    "          • thumb: [carpal, MCP, IP, tip]\n"
    "          • index: [MCP, PIP, DIP, tip]\n"
    "          • middle: [MCP, PIP, DIP, tip]\n"
    "          • ring: [MCP, PIP, DIP, tip]\n"
    "          • little: [MCP, PIP, DIP, tip]\n"
    "          • palm_base: [x,y,z]\n"
    "      - facial_analysis: {eyebrow_height: float, eye_closure: [left,right], mouth_shape: str, jaw_tension: float, cheek_definition: str}\n"
    "3. Use EXACT timing boundaries (0.0-0.1s, 0.1-0.2s, ..., 0.9-1.0s)\n\n"
    "DERIVATION RULES:\n"
    "• hand_shape: Determined by finger joint angles (MCP/PIP/DIP flexion) and thumb opposition\n"
    "• location: Calculated from wrist positions relative to nose-chin-chest landmarks\n"
    "• movement: Vector analysis of wrist trajectory between segments\n"
    "• dominant_hand: Hand with greatest displacement amplitude\n"
    "• facial_expression: Synthesized from eyebrow/eye/mouth metrics\n\n"
    "OUTPUT FORMAT: A COMPACT JSON ARRAY (MINIMAL WHITESPACE) FOLLOWING THIS STRUCTURE:\n"
    """[{"segment":1,"start":0.0,"end":0.1,"hand_shape":"open/relaxed","location":"at-sides","movement":"stationary","dominant_hand":"N/A","facial_expression":"neutral","detailed_observations":{"head_neck":{"nose_position":[0.51,0.42,0.0],"eye_aperture":0.9,"ear_visibility":false},"upper_body":{"shoulder_angle":175.2,"elbow_flexion":[178.1,177.9],"wrist_position":[[0.22,0.75],[0.78,0.76]]},"hand_joints":{"left":{"thumb":[[0.21,0.74],[0.22,0.73],[0.23,0.72],[0.24,0.71]],"index":[[0.20,0.75],[0.21,0.74],[0.22,0.73],[0.23,0.72]],"middle":[[0.19,0.76],[0.20,0.75],[0.21,0.74],[0.22,0.73]],"ring":[[0.18,0.77],[0.19,0.76],[0.20,0.75],[0.21,0.74]],"little":[[0.17,0.78],[0.18,0.77],[0.19,0.76],[0.20,0.75]],"palm_base":[0.20,0.75,0.0]},"right":{"thumb":[[0.79,0.74],[0.78,0.73],[0.77,0.72],[0.76,0.71]],"index":[[0.80,0.75],[0.79,0.74],[0.78,0.73],[0.77,0.72]],"middle":[[0.81,0.76],[0.80,0.75],[0.79,0.74],[0.78,0.73]],"ring":[[0.82,0.77],[0.81,0.76],[0.80,0.75],[0.79,0.74]],"little":[[0.83,0.78],[0.82,0.77],[0.81,0.76],[0.80,0.75]],"palm_base":[0.80,0.75,0.0]}},"facial_analysis":{"eyebrow_height":0.2,"eye_closure":[0.95,0.94],"mouth_shape":"relaxed","jaw_tension":0.1,"cheek_definition":"smooth"}},...]"""
)

def describe_sign(video_path: Path) -> list:
    """Send video to Gemini API with robust JSON extraction"""
    # Get MIME type and encode video
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type:
        mime_type = "video/mp4"
    
    with open(video_path, "rb") as f:
        video_data = f.read()
        if len(video_data) > 20 * 1024 * 1024:  # 20MB limit
            logger.warning(f"Video too large ({len(video_data)/1024/1024:.1f}MB), skipping: {video_path.name}")
            return None
        encoded_video = base64.b64encode(video_data).decode("utf-8")

    # API configuration
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}
    
    # Max token limit
    generation_config = {
        "temperature": 0,
        "max_output_tokens": 8192,  # Maximum for Flash models
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
            logger.info(f"Sending request for {video_path.name} (attempt {attempt+1}/{MAX_RETRIES})")
            response = requests.post(url, json=payload, headers=headers, params=params, timeout=120)
            response.raise_for_status()
            
            # Parse response JSON
            response_json = response.json()
            
            # Check for API errors
            if "error" in response_json:
                error_msg = response_json["error"].get("message", "Unknown API error")
                logger.error(f"API Error: {error_msg}")
                return None
                
            # Validate response structure
            if "candidates" not in response_json or not response_json["candidates"]:
                logger.error("No candidates in response")
                return None
                
            candidate = response_json["candidates"][0]
            if "content" not in candidate or "parts" not in candidate["content"]:
                logger.error("Invalid candidate structure")
                return None
                
            parts = candidate["content"]["parts"]
            if not parts or "text" not in parts[0]:
                logger.error("No text in response parts")
                return None
                
            content = parts[0]["text"]
            logger.debug(f"Received response: {content[:200]}...")
            
            # Extract JSON using regex
            json_match = re.search(r'\[[\s\S]*\]', content)
            if not json_match:
                logger.error(f"JSON not found in response. First 200 chars: {content[:200]}")
                return None
                
            json_str = json_match.group()
            
            # Attempt to fix common truncation issues
            json_str = fix_truncated_json(json_str)
            
            return json.loads(json_str)
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else "Unknown"
            if status_code == 503 and attempt < MAX_RETRIES - 1:
                jitter = random.uniform(0.5, 1.5)
                sleep_time = min(delay * jitter, MAX_DELAY)
                logger.warning(f"Service Unavailable (503). Retry #{attempt+1} in {sleep_time:.1f}s")
                time.sleep(sleep_time)
                delay *= 2
                continue
            else:
                logger.error(f"HTTP Error {status_code}: {str(e)}")
                if hasattr(e, 'response') and e.response.text:
                    logger.debug(f"Response snippet: {e.response.text[:500]}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {str(e)}")
            if 'json_str' in locals():
                logger.debug(f"JSON snippet: {json_str[:500]}")
                save_debug_file(video_path, json_str)
            return None
        except Exception as e:
            logger.error(f"Unexpected Error: {type(e).__name__} - {str(e)}")
            return None

    return None

def fix_truncated_json(json_str: str) -> str:
    """Attempt to repair truncated JSON responses"""
    # Count open brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    
    # If missing closing braces, add them
    if open_braces > close_braces:
        needed = open_braces - close_braces
        json_str += '}' * needed
        logger.info(f"Added {needed} closing braces to JSON")
    
    # Check if array is closed
    if not json_str.strip().endswith(']'):
        json_str = json_str.rstrip() + ']'
        logger.info("Added closing array bracket")
    
    return json_str

def save_debug_file(video_path: Path, content: str):
    """Save problematic responses for debugging"""
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)
    
    debug_file = debug_dir / f"{video_path.stem}_response.txt"
    with open(debug_file, "w") as f:
        f.write(content)
    
    logger.info(f"Saved debug response to {debug_file}")

def generate_vocabulary():
    """Process all videos in directory and save results"""
    vocabulary = []
    video_dir = Path(VIDEO_DIR)

    if not video_dir.exists():
        logger.error(f"Directory not found: {video_dir}")
        return

    # Process video files
    video_files = list(video_dir.glob("*.*"))
    logger.info(f"Found {len(video_files)} files in directory")
    
    for i, video_path in enumerate(video_files):
        if video_path.suffix.lower() not in [".mp4", ".mov", ".avi"]:
            logger.info(f"Skipping non-video file: {video_path.name}")
            continue

        word = video_path.stem
        logger.info(f"\nProcessing ({i+1}/{len(video_files)}): {word}")

        start_time = time.time()
        description = describe_sign(video_path)
        process_time = time.time() - start_time

        if description:
            vocabulary.append({
                "word": word,
                "description": description,
                "video_file": video_path.name,
                "processing_time": round(process_time, 1)
            })
            logger.info(f"Success: {len(description)} segments in {process_time:.1f}s")
        else:
            logger.error(f"Failed to process {word}")

        # Delay between requests
        time.sleep(3)

    # Save results
    if vocabulary:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(vocabulary, f, indent=2)
        logger.info(f"\nSuccess! Saved {len(vocabulary)} signs to {OUTPUT_JSON}")
        logger.info(f"Total words processed: {len(vocabulary)}/{len(video_files)}")
    else:
        logger.error("\nNo valid sign descriptions generated")

if __name__ == "__main__":
    logger.info(f"Starting PSL vocabulary generation with {MODEL_NAME}")
    generate_vocabulary()