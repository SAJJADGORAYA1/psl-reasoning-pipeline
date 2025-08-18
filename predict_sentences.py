import json
from pathlib import Path
import time
import sys
import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
VOCAB_BANK = "psl_vocabulary_bank.json"
MOTION_THRESHOLD = 15         # Pixel difference threshold for sign boundaries
MIN_SIGN_DURATION = 0.2       # Minimum sign duration (seconds)
MAX_SIGN_DURATION = 2.0       # Maximum sign duration (seconds)
CONFIDENCE_THRESHOLD = 0.55   # Minimum similarity score to accept prediction
LK_PARAMS = {
    'winSize': (15, 15),
    'maxLevel': 2,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    'flags': 0,
    'minEigThreshold': 1e-4
}

def fixed_interval_segmentation(video_path: Path, segment_length=1.0):
    """Segment video into fixed intervals"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    segments = []
    segment_frames = int(fps * segment_length)
    
    for start_frame in range(0, total_frames, segment_frames):
        end_frame = min(start_frame + segment_frames, total_frames)
        duration = (end_frame - start_frame) / fps
        
        segments.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_frame / fps,
            "end_time": end_frame / fps,
            "duration": duration
        })
    
    cap.release()
    print(f"✂️ Fixed segmentation: {len(segments)} segments")
    return segments, fps
def extract_motion_features(segment_path: Path):
    """Extract trajectory features from sign segment using optical flow"""
    cap = cv2.VideoCapture(str(segment_path))
    if not cap.isOpened():
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255
    
    # Feature detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    
    if p0 is None:
        cap.release()
        return {"error": "No features detected"}
    
    trajectory = []
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, frame_gray, p0, None, **LK_PARAMS
        )
        
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            # Calculate motion vectors
            if len(good_new) > 0:
                motion_vectors = good_new - good_old
                avg_motion = np.mean(motion_vectors, axis=0)
                trajectory.append(avg_motion)
            
            # Update for next frame
            prev_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        
        frame_index += 1
    
    cap.release()
    
    # Create motion signature
    if len(trajectory) == 0:
        return {"error": "No motion detected"}
    
    trajectory = np.array(trajectory)
    return {
        "motion_signature": trajectory.flatten().tolist(),
        "frame_count": frame_index,
        "feature_points": len(p0)
    }

def extract_video_segment(video_path: Path, start_frame: int, end_frame: int, fps: float):
    """Extract a video segment as temporary file"""
    if start_frame >= end_frame:
        raise ValueError(f"Invalid frame range: {start_frame}-{end_frame}")
        
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temp directory
    temp_dir = Path("temp_segments")
    temp_dir.mkdir(exist_ok=True)
    
    temp_file = temp_dir / f"segment_{start_frame}_{end_frame}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_file), fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Validate extraction
    seg_cap = cv2.VideoCapture(str(temp_file))
    seg_frames = int(seg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seg_cap.release()
    
    if seg_frames == 0:
        temp_file.unlink()
        return None
        
    return temp_file

def predict_sign(segment_path: Path, vocab_data: list):
    """Predict sign by comparing motion signatures"""
    # Extract motion features from segment
    test_features = extract_motion_features(segment_path)
    
    if "error" in test_features:
        print(f"   Feature error: {test_features['error']}")
        return None
    
    # Find best match in vocabulary
    best_match = None
    best_score = -1
    best_word = None
    
    test_signature = np.array(test_features["motion_signature"]).reshape(1, -1)
    
    for entry in vocab_data:
        if "motion_signature" not in entry:
            continue
            
        # Convert and reshape vocabulary signature
        vocab_sig = np.array(entry["motion_signature"]).reshape(1, -1)
        
        # Handle different lengths by using the min length
        min_length = min(test_signature.shape[1], vocab_sig.shape[1])
        test_trimmed = test_signature[:, :min_length]
        vocab_trimmed = vocab_sig[:, :min_length]
        
        # Calculate cosine similarity
        similarity = cosine_similarity(test_trimmed, vocab_trimmed)[0][0]
        
        if similarity > best_score:
            best_score = similarity
            best_match = entry
            best_word = entry["word"]
    
    if best_match:
        return {
            "predicted_word": best_word,
            "confidence_score": best_score,
            "test_features": test_features
        }
    return None

def predict_sentence(test_video_path: Path, vocab_data: list):
    """Predict sentence with motion-based segmentation"""
    try:
        print(f"\n Starting sentence prediction: {test_video_path.name}")
        segments, fps = fixed_interval_segmentation(test_video_path, segment_length=1.5)
        
        if not segments:
            print(" No signs detected in video")
            return None
        
        predicted_words = []
        
        for i, seg in enumerate(segments):
            print(f"\n Processing sign {i+1}: "
                  f"{seg['start_time']:.1f}s to {seg['end_time']:.1f}s "
                  f"({seg['duration']:.2f}s)")
            
            # Extract segment
            seg_video = extract_video_segment(
                test_video_path, 
                seg['start_frame'], 
                seg['end_frame'], 
                fps
            )
            
            if not seg_video or not seg_video.exists():
                print("    Segment extraction failed")
                predicted_words.append("[UNKNOWN]")
                continue
                
            # Predict sign
            result = predict_sign(seg_video, vocab_data)
            seg_video.unlink()  # Cleanup
            
            if result:
                confidence = result['confidence_score']
                word = result['predicted_word']
                
                print(f"    Predicted: '{word}' (Confidence: {confidence:.2f})")
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_words.append(word)
                else:
                    predicted_words.append(f"[{word}?]")  # Mark low confidence
            else:
                print("    Prediction failed")
                predicted_words.append("[UNKNOWN]")
        
        return " ".join(predicted_words)
    
    except Exception as e:
        print(f" Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def build_vocabulary(video_dir: Path):
    """Build vocabulary with motion signatures from sign videos"""
    print(f"🏗️ Building vocabulary from: {video_dir}")
    vocab = []
    
    if not video_dir.exists():
        print(f" Directory not found: {video_dir}")
        return None
    
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    
    if not video_files:
        print(" No video files found")
        return None
    
    for video_file in video_files:
        word = video_file.stem.split('_')[0].lower()  # Extract base word
        print(f"Processing: {video_file.name} -> '{word}'")
        
        # Extract motion features
        features = extract_motion_features(video_file)
        
        if "error" in features:
            print(f"   Skipped - {features['error']}")
            continue
            
        vocab.append({
            "word": word,
            "motion_signature": features["motion_signature"],
            "source_video": video_file.name,
            "feature_points": features.get("feature_points", 0),
            "frames": features.get("frame_count", 0)
        })
    
    # Save vocabulary
    with open(VOCAB_BANK, "w") as f:
        json.dump(vocab, f, indent=2)
    
    print(f" Vocabulary built with {len(vocab)} entries")
    return vocab

def load_vocabulary():
    """Load existing vocabulary bank"""
    try:
        with open(VOCAB_BANK, "r") as f:
            vocab_data = json.load(f)
        print(f" Loaded vocabulary bank with {len(vocab_data)} entries")
        return vocab_data
    except FileNotFoundError:
        print(" Vocabulary bank not found. Please build it first.")
        return None
    except Exception as e:
        print(f" Error loading vocabulary: {str(e)}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_sentences.py <command> [options]")
        print("Commands:")
        print("  build_vocab <path_to_vocabulary_videos>  - Build motion vocabulary")
        print("  predict <path_to_sentence_video.mp4>      - Predict sentence")
        return
        
    command = sys.argv[1]
    
    if command == "build_vocab":
        if len(sys.argv) < 3:
            print("Please provide path to vocabulary videos directory")
            return
        vocab_dir = Path(sys.argv[2])
        build_vocabulary(vocab_dir)
        
    elif command == "predict":
        if len(sys.argv) < 3:
            print("Please provide path to a sentence video file")
            return
            
        test_video_path = Path(sys.argv[2])
        if not test_video_path.exists():
            print(f" Video not found: {test_video_path}")
            return
        
        vocab_data = load_vocabulary()
        if not vocab_data:
            return
            
        start_time = time.time()
        sentence = predict_sentence(test_video_path, vocab_data)
        
        if sentence:
            print("\n" + "="*50)
            print(f" PREDICTED SENTENCE: {sentence}")
            print("="*50)
            print(f" Total processing time: {time.time() - start_time:.2f}s")
            
            # Save results
            result = {
                "input_video": test_video_path.name,
                "predicted_sentence": sentence,
                "processing_time": time.time() - start_time
            }
            output_file = f"sentence_result_{test_video_path.stem}.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved results to '{output_file}'")
        else:
            print("\n Failed to predict sentence")
            
    else:
        print(f" Unknown command: {command}")

if __name__ == "__main__":
    main()