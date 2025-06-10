


from flask import Flask, request, jsonify

import easyocr
from fuzzywuzzy import fuzz
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask_cors import CORS
import io

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes



reader = easyocr.Reader(['en', 'ar'])


model = models.resnet50(pretrained=True)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_bad_words(filepath="bad_words.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

BAD_WORDS = load_bad_words()


def detect_text_and_check(image_data, bad_words, similarity_threshold=80):
    """Detects text in image and checks for bad words"""
    try:
        
        import numpy as np
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        results = reader.readtext(image_np, detail=1)
        extracted_text = " ".join([res[1] for res in results])

        detected_bad_words = [
            bad_word for bad_word in bad_words
            if fuzz.partial_ratio(bad_word.lower(), extracted_text.lower()) > similarity_threshold
        ]

        return {
            "success": True,
            "detected_bad_words": detected_bad_words,
            "has_bad_words": len(detected_bad_words) > 0
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def classify_image(image_data):
    """Classifies image content and checks for inappropriate content"""
    try:
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_tensor = transform(img)
        batch_tensor = torch.unsqueeze(img_tensor, 0)
        
        with torch.no_grad():
            output = model(batch_tensor)
        
        
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, indices = torch.sort(output, descending=True)
        
        
        try:
            import urllib.request
            classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            with urllib.request.urlopen(classes_url) as f:
                classes = [line.strip() for line in f.read().decode('utf-8').splitlines()]
        except:
            
            classes = [str(i) for i in range(1000)]
        
        
        top_results = []
        for idx in indices[0][:5]:
            class_name = classes[idx]
            confidence = probabilities[idx].item() * 100
            top_results.append({"class": class_name, "confidence": confidence})
        
        
        inappropriate_keywords = ['naked', 'underwear', 'bikini', 'gun', 'weapon']
        inappropriate_classes = []
        
        for result in top_results:
            class_name = result["class"].lower()
            confidence = result["confidence"]
            if any(keyword in class_name for keyword in inappropriate_keywords) and confidence > 20:
                inappropriate_classes.append({
                    "class": result["class"],
                    "confidence": confidence
                })
        
        return {
            "success": True,
            "has_inappropriate_content": len(inappropriate_classes) > 0
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]

    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        return jsonify({"error": "Unsupported file format"}), 400

    image_data = image.read()

    text_result = detect_text_and_check(image_data, BAD_WORDS)
    image_result = classify_image(image_data)

    result = {
        "overall_status": "rejected" if (
            text_result.get("has_bad_words", False) or 
            image_result.get("has_inappropriate_content", False)
        ) else "accepted",
        "text_analysis_success": text_result.get("success", False)
    }

    if text_result.get("has_bad_words", False):
        result["bad_words_detected"] = text_result.get("detected_bad_words", [])

    return jsonify(result)
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)