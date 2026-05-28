"""
Aksha_AI — Flask Backend
Image upload + TensorFlow skin disease prediction API
"""

import os
import uuid
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import logging

# ── Config ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])  # In production, restrict to your frontend domain

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_PATH"] = "models/aksha_skin_model.h5"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "webp"}
app.config["IMG_SIZE"] = (224, 224)  # MobileNetV2 input size

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Disease Classes ───────────────────────────────────────────────────────────
# These must match the order used during model training
DISEASE_CLASSES = [
    "Acne & Pimples",
    "Allergic Reaction",
    "Contact Dermatitis",
    "Dry Skin",
    "Eczema",
    "Fungal Infection",
    "Normal Skin",
    "Psoriasis",
    "Rosacea",
    "Skin Rash",
    "Urticaria (Hives)",
    "Vitiligo",
]

DISEASE_META = {
    "Acne & Pimples": {
        "severity": "mild",
        "description": "Inflammatory acne including blackheads, whiteheads, and cystic breakouts.",
        "recommendation": "Use gentle, non-comedogenic skincare. Consult a dermatologist if severe.",
        "color": "#d85a30",
    },
    "Allergic Reaction": {
        "severity": "moderate",
        "description": "Skin reaction triggered by allergens causing redness and swelling.",
        "recommendation": "Identify and avoid triggers. Antihistamines may provide relief.",
        "color": "#378add",
    },
    "Contact Dermatitis": {
        "severity": "moderate",
        "description": "Skin irritation caused by direct contact with an irritant or allergen.",
        "recommendation": "Avoid the irritant. Topical corticosteroids may help reduce inflammation.",
        "color": "#ba7517",
    },
    "Dry Skin": {
        "severity": "mild",
        "description": "Abnormal skin dryness, flaking, and tightness.",
        "recommendation": "Use a rich moisturiser regularly. Drink adequate water.",
        "color": "#639922",
    },
    "Eczema": {
        "severity": "moderate",
        "description": "Atopic dermatitis causing chronic itchy, inflamed skin patches.",
        "recommendation": "Moisturise frequently. Avoid triggers. Consult a dermatologist.",
        "color": "#993556",
    },
    "Fungal Infection": {
        "severity": "moderate",
        "description": "Tinea or ringworm fungal infection on facial skin.",
        "recommendation": "Antifungal creams. Keep skin dry and clean. See a doctor if spreading.",
        "color": "#3b6d11",
    },
    "Normal Skin": {
        "severity": "none",
        "description": "No significant skin condition detected.",
        "recommendation": "Maintain your current skincare routine. Regular sunscreen use recommended.",
        "color": "#1d9e75",
    },
    "Psoriasis": {
        "severity": "high",
        "description": "Chronic autoimmune condition causing red, scaly skin patches.",
        "recommendation": "Consult a dermatologist. Treatment may include topicals or phototherapy.",
        "color": "#534ab7",
    },
    "Rosacea": {
        "severity": "moderate",
        "description": "Chronic redness and visible blood vessels mainly on the face.",
        "recommendation": "Avoid triggers (alcohol, spicy food, sun). Consult a dermatologist.",
        "color": "#993c1d",
    },
    "Skin Rash": {
        "severity": "mild",
        "description": "General rash or skin irritation with redness and inflammation.",
        "recommendation": "Identify cause. Hydrocortisone cream for mild cases. See a doctor if persistent.",
        "color": "#ba7517",
    },
    "Urticaria (Hives)": {
        "severity": "moderate",
        "description": "Raised, itchy welts on the skin triggered by allergic reactions.",
        "recommendation": "Antihistamines for relief. Identify and avoid the trigger allergen.",
        "color": "#185fa5",
    },
    "Vitiligo": {
        "severity": "low",
        "description": "Loss of skin pigmentation resulting in white patches.",
        "recommendation": "Consult a dermatologist. Sun protection is essential for affected areas.",
        "color": "#444441",
    },
}

# ── Model Loader ──────────────────────────────────────────────────────────────

_model = None  # lazy-loaded singleton


def build_model(num_classes: int) -> Model:
    """Build MobileNetV2-based transfer learning model for skin classification."""
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    # Freeze base layers for transfer learning
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=predictions)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_model() -> Model:
    """Return cached model, loading or building as needed."""
    global _model
    if _model is not None:
        return _model

    model_path = app.config["MODEL_PATH"]
    if os.path.exists(model_path):
        logger.info(f"Loading trained model from {model_path}")
        _model = load_model(model_path)
    else:
        logger.warning(
            "No trained model found — building untrained base model. "
            "Train the model with your dataset before using in production."
        )
        _model = build_model(num_classes=len(DISEASE_CLASSES))

    return _model


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in app.config["ALLOWED_EXTENSIONS"]


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize, normalise and batch an image for MobileNetV2."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(app.config["IMG_SIZE"], Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)       # scale to [-1, 1]
    return np.expand_dims(arr, axis=0)  # add batch dimension → (1, 224, 224, 3)


def build_prediction_response(probabilities: np.ndarray) -> dict:
    """Turn raw softmax probabilities into a structured prediction response."""
    scores = probabilities[0]
    indices_sorted = np.argsort(scores)[::-1]

    top_index = int(indices_sorted[0])
    top_class = DISEASE_CLASSES[top_index]
    top_conf = float(scores[top_index])

    all_results = [
        {
            "condition": DISEASE_CLASSES[i],
            "confidence": round(float(scores[i]) * 100, 2),
            "meta": DISEASE_META.get(DISEASE_CLASSES[i], {}),
        }
        for i in indices_sorted
    ]

    # Top-3 for display
    top3 = all_results[:3]

    meta = DISEASE_META.get(top_class, {})

    return {
        "status": "success",
        "primary": {
            "condition": top_class,
            "confidence": round(top_conf * 100, 2),
            "severity": meta.get("severity", "unknown"),
            "description": meta.get("description", ""),
            "recommendation": meta.get("recommendation", ""),
            "color": meta.get("color", "#1d9e75"),
        },
        "top3": top3,
        "all_results": all_results,
        "disclaimer": (
            "This is an AI-generated screening result and NOT a medical diagnosis. "
            "Always consult a qualified dermatologist for proper evaluation."
        ),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({"app": "Aksha_AI API", "status": "running", "version": "1.0.0"})


@app.route("/api/health", methods=["GET"])
def health_check():
    """Lightweight health-check endpoint for monitoring."""
    return jsonify({"status": "healthy", "model_loaded": _model is not None})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Accepts:
        - multipart/form-data  →  field name: 'image'
        - application/json     →  { "image": "<base64-encoded-string>" }

    Returns:
        JSON with primary prediction, top-3, all results, and metadata.
    """
    image_bytes = None

    # ── Accept multipart file upload ─────────────────────────────────────────
    if "image" in request.files:
        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "File type not allowed. Use JPG, PNG, or WEBP"}), 400

        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        image_bytes = open(save_path, "rb").read()

    # ── Accept base64-encoded image in JSON body ──────────────────────────────
    elif request.is_json and "image" in request.json:
        try:
            b64_data = request.json["image"]
            # Strip data URL prefix if present: "data:image/jpeg;base64,..."
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
        except Exception as exc:
            return jsonify({"status": "error", "message": f"Invalid base64 image: {exc}"}), 400

    else:
        return jsonify({
            "status": "error",
            "message": "No image provided. Send 'image' as a file upload or base64 JSON field.",
        }), 400

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        model = get_model()
        tensor = preprocess_image(image_bytes)
        probs = model.predict(tensor, verbose=0)
        response = build_prediction_response(probs)
        return jsonify(response), 200

    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"status": "error", "message": f"Prediction failed: {str(exc)}"}), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction endpoint — accepts up to 5 images at once.

    Accepts multipart/form-data with multiple 'images[]' fields.
    """
    files = request.files.getlist("images[]")
    if not files:
        return jsonify({"status": "error", "message": "No images provided"}), 400
    if len(files) > 5:
        return jsonify({"status": "error", "message": "Maximum 5 images per batch"}), 400

    model = get_model()
    results = []

    for file in files:
        if not allowed_file(file.filename):
            results.append({"file": file.filename, "status": "error", "message": "Unsupported file type"})
            continue
        try:
            image_bytes = file.read()
            tensor = preprocess_image(image_bytes)
            probs = model.predict(tensor, verbose=0)
            result = build_prediction_response(probs)
            result["file"] = secure_filename(file.filename)
            results.append(result)
        except Exception as exc:
            results.append({"file": file.filename, "status": "error", "message": str(exc)})

    return jsonify({"status": "success", "results": results, "count": len(results)}), 200


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Return all detectable disease classes with metadata."""
    return jsonify({
        "status": "success",
        "classes": [
            {"name": cls, "meta": DISEASE_META.get(cls, {})}
            for cls in DISEASE_CLASSES
        ],
        "total": len(DISEASE_CLASSES),
    })


@app.route("/api/model/info", methods=["GET"])
def model_info():
    """Return model architecture details."""
    model = get_model()
    return jsonify({
        "status": "success",
        "architecture": "MobileNetV2 (transfer learning)",
        "input_shape": list(app.config["IMG_SIZE"]) + [3],
        "num_classes": len(DISEASE_CLASSES),
        "total_params": model.count_params(),
        "trainable_params": sum(
            tf.size(w).numpy() for w in model.trainable_weights
        ),
    })


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Aksha_AI backend on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
