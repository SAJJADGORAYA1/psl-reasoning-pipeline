import requests
import json
from pathlib import Path
import time
import mimetypes
import base64
import random
import math
from datetime import datetime

# Configuration
API_KEY = "AIzaSyBNsavVonNcYikKM0hTzPwtehVmFjPLJZo"
MODEL_NAME = "gemini-2.0-flash"  # Consider switching to "gemini-1.5-flash-latest" for higher limits
VIDEO_DIR = "Words"
OUTPUT_JSON = "psl_vocabulary_bank.json"
ERROR_LOG = "processing_errors.log"
MAX_RETRIES = 8  # Increased retries for rate limits
INITIAL_DELAY = 1
MAX_DELAY = 60
MIN_FILE_SIZE = 1024

# Rate limiting configuration
RATE_LIMIT_WINDOW = 60  # 60-second window
MAX_REQUESTS_PER_MINUTE = 15  # Gemini's default rate limit for free tier
MIN_INTER_REQUEST_DELAY = 3.0  # Minimum delay between requests (seconds)

# Global rate limiting state
request_timestamps = []

# Prompt remains the same
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

def rate_limit_enforce():
    """Enforce rate limiting based on API constraints"""
    global request_timestamps
    
    # Clean up old timestamps
    now = time.time()
    request_timestamps = [t for t in request_timestamps if now - t < RATE_LIMIT_WINDOW]
    
    # Check if we've reached the rate limit
    if len(request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
        # Calculate when we can make the next request
        oldest = request_timestamps[0]
        wait_time = max(RATE_LIMIT_WINDOW - (now - oldest), 0) + 1
        print(f"⏳ Rate limit reached. Waiting {wait_time:.1f} seconds...")
        time.sleep(wait_time)
        # Reset after waiting
        request_timestamps = []
    
    # Ensure minimum delay between requests
    if request_timestamps:
        last_request = request_timestamps[-1]
        elapsed = now - last_request
        if elapsed < MIN_INTER_REQUEST_DELAY:
            wait_time = MIN_INTER_REQUEST_DELAY - elapsed
            print(f"⏳ Enforcing inter-request delay: {wait_time:.1f}s")
            time.sleep(wait_time)
    
    # Record new request
    request_timestamps.append(time.time())

def log_error(message):
    """Log errors to a file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def describe_sign(video_path: Path) -> dict:
    """Send video to Gemini API with enhanced rate limiting"""
    # Validate file size
    if video_path.stat().st_size < MIN_FILE_SIZE:
        msg = f"⚠️ Skipping small file: {video_path.name} ({video_path.stat().st_size} bytes)"
        print(msg)
        log_error(msg)
        return None

    # Get MIME type and encode video
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type:
        mime_type = "video/mp4"
    
    try:
        with open(video_path, "rb") as f:
            encoded_video = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        msg = f"🚨 File read error: {str(e)}"
        print(msg)
        log_error(msg)
        return None

    # API configuration
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}
    
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
            # Enforce rate limiting before each request
            rate_limit_enforce()
            
            response = requests.post(url, json=payload, headers=headers, params=params)
            response.raise_for_status()
            
            # Parse JSON response
            response_json = response.json()
            
            # Validate response structure
            if "candidates" not in response_json:
                if "error" in response_json:
                    error_msg = response_json["error"].get("message", "Unknown API error")
                    raise ValueError(f"API Error: {error_msg}")
                raise KeyError("Missing 'candidates' in response")
            
            # Extract content text
            content = response_json["candidates"][0]["content"]["parts"][0]["text"]
            cleaned_content = content.strip().replace("```json", "").replace("```", "")
            
            # Validate JSON format
            if not cleaned_content.startswith("{"):
                raise ValueError("Response not JSON formatted")
            
            # Parse and validate content
            parsed = json.loads(cleaned_content)
            
            # Verify required fields
            required_fields = [
                "hands_used", "primary_hand", "two_hand_relation", 
                "contact_presence", "body_region_primary", "path_family",
                "arm_motion", "hand_motion", "finger_motion", 
                "movement_summary", "overall_description"
            ]
            
            if not all(field in parsed for field in required_fields):
                missing = [f for f in required_fields if f not in parsed]
                raise KeyError(f"Missing fields: {', '.join(missing)}")
                
            return parsed
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "Unknown"
            
            # Handle rate limiting specifically
            if status_code == 429:
                # Try to get Retry-After header
                retry_after = e.response.headers.get('Retry-After', None)
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                    except ValueError:
                        wait_time = min(delay * 2, MAX_DELAY)
                else:
                    wait_time = min(delay * (2 ** attempt), MAX_DELAY)
                
                print(f"⚠️ Rate limited (429). Waiting {wait_time}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                delay = wait_time * 2  # Double for next potential retry
                continue
            
            if status_code == 503 and attempt < MAX_RETRIES - 1:
                # Exponential backoff with jitter
                jitter = random.uniform(0.5, 1.5)
                sleep_time = min(delay * jitter, MAX_DELAY)
                print(f"⚡ Model overloaded. Retry #{attempt+1} in {sleep_time:.1f}s")
                time.sleep(sleep_time)
                delay *= 2
            else:
                msg = f"🚨 HTTP Error {status_code}: {str(e)}"
                print(msg)
                log_error(msg)
                if e.response and e.response.text:
                    error_snippet = e.response.text[:200]
                    print(f"Response: {error_snippet}...")
                    log_error(f"Response snippet: {error_snippet}")
                return None
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            error_type = type(e).__name__
            msg = f"⚠️ Parsing error ({error_type}): {str(e)}"
            print(msg)
            log_error(msg)
            if 'cleaned_content' in locals():
                snippet = cleaned_content[:200] if cleaned_content else "No content"
                print(f"Response snippet: {snippet}...")
                log_error(f"Response snippet: {snippet}")
            if attempt < MAX_RETRIES - 1:
                jitter = random.uniform(0.5, 1.5)
                sleep_time = min(INITIAL_DELAY * (2 ** attempt) * jitter, MAX_DELAY)
                print(f"🔄 Retry #{attempt+1} in {sleep_time:.1f}s")
                time.sleep(sleep_time)
            else:
                msg = f"❌ Max retries exceeded for {video_path.name}"
                print(msg)
                log_error(msg)
                return None
        except Exception as e:
            msg = f"🚨 Unexpected error: {type(e).__name__} - {str(e)}"
            print(msg)
            log_error(msg)
            return None

    return None

def generate_vocabulary():
    # Create directory if needed
    video_dir = Path(VIDEO_DIR)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize error log
    with open(ERROR_LOG, "w") as f:
        f.write(f"PSL Vocabulary Processing Log - {datetime.now()}\n")
        f.write("="*50 + "\n")
    
    # Load existing data if available
    processed_words = set()
    vocabulary = []
    failed_words = []
    
    output_path = Path(OUTPUT_JSON)
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                vocabulary = json.load(f)
            processed_words = {entry["word"] for entry in vocabulary}
            print(f"📖 Resuming: Found {len(vocabulary)} processed words")
        except Exception as e:
            msg = f"⚠️ Error loading existing JSON: {str(e)} - Starting fresh"
            print(msg)
            log_error(msg)

    # Process video files
    video_files = [f for f in video_dir.iterdir() if f.is_file()]
    total_count = len(video_files)
    
    for i, video_path in enumerate(video_files):
        if video_path.suffix.lower() not in [".mp4", ".mov", ".avi"]:
            continue
            
        word = video_path.stem
        if word in processed_words:
            print(f"⏭️ Skipping already processed: {word} ({i+1}/{total_count})")
            continue

        print(f"\n🔍 Processing ({i+1}/{total_count}): {word}")
        
        # Attempt to get description
        description = describe_sign(video_path)
        
        if description:
            vocabulary.append({
                "word": word,
                "description": description,
                "video_file": video_path.name
            })
            print(f"✅ Success: {word}")
            
            # Save after each successful processing
            with open(OUTPUT_JSON, "w") as f:
                json.dump(vocabulary, f, indent=2)
                print(f"💾 Auto-saved progress")
        else:
            failed_words.append(word)
            msg = f"❌ Failed to process {word} after {MAX_RETRIES} attempts"
            print(msg)
            log_error(msg)
        
        # Add delay between videos to prevent rate limiting
        inter_video_delay = random.uniform(2.0, 5.0)
        print(f"⏳ Waiting {inter_video_delay:.1f}s before next video")
        time.sleep(inter_video_delay)

    # Final save and report
    if vocabulary:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(vocabulary, f, indent=2)
        print(f"\n🎉 Success! Saved {len(vocabulary)} sign descriptions to {OUTPUT_JSON}")
    
    if failed_words:
        print(f"\n⚠️ Failed to process {len(failed_words)} words:")
        for word in failed_words:
            print(f"  - {word}")
        print(f"See {ERROR_LOG} for details")
    
    print(f"\nProcessing complete. Total: {total_count} videos, Success: {len(vocabulary)}, Failed: {len(failed_words)}")

if __name__ == "__main__":
    print(f"Starting PSL vocabulary processing at {datetime.now()}")
    print(f"Rate limit configuration: Max {MAX_REQUESTS_PER_MINUTE} requests per minute")
    generate_vocabulary()