from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge


ROOT = Path(__file__).resolve().parent
ALLOWED_FORMATS = {"BMP", "JPEG", "PNG", "TIFF", "WEBP"}
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_BYTES = 12 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _clamp_score(value: float) -> int:
    return int(round(max(0.0, min(100.0, value))))


def _metric(score: int | None, raw_value: float | None, unit: str) -> dict[str, Any]:
    if score is None:
        return {"score": None, "status": "Unavailable", "raw_value": None, "unit": unit}
    status = "Good" if score >= 75 else "Review" if score >= 50 else "Poor"
    return {
        "score": score,
        "status": status,
        "raw_value": round(raw_value, 3) if raw_value is not None else None,
        "unit": unit,
    }


def analyze_image(image: np.ndarray) -> dict[str, Any]:
    """Return notebook-based image signals with explicit, uncalibrated proxy scores."""
    gray = _gray(image)
    height, width = gray.shape

    blur_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = _clamp_score(100 * math.log1p(blur_raw) / math.log1p(500))

    scale = min(1.0, 1024 / max(height, width))
    if scale < 1:
        analysis_gray = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        analysis_gray = gray

    edges = cv2.Canny(analysis_gray, 50, 150)
    min_line_length = max(20, int(0.3 * analysis_gray.shape[1]))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=10,
    )
    skew_degrees: float | None = None
    if lines is not None and len(lines) >= 3:
        segments = lines.reshape(-1, 4)
        deviations = []
        for x1, y1, x2, y2 in segments:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
            deviation = angle % 90
            deviations.append(min(deviation, 90 - deviation))
        skew_degrees = float(np.median(deviations))
    skew_score = (
        _clamp_score(100 - skew_degrees * 12) if skew_degrees is not None else None
    )

    fft_gray = analysis_gray.astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(fft_gray)))
    center_y, center_x = spectrum.shape[0] // 2, spectrum.shape[1] // 2
    y, x = np.ogrid[: spectrum.shape[0], : spectrum.shape[1]]
    radius = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    radius_max = min(center_y, center_x)
    if radius_max:
        high_frequency_ratio = float(
            spectrum[radius > 0.5 * radius_max].sum()
            / (spectrum.sum() + 1e-9)
        )
    else:
        high_frequency_ratio = 0.0
    resolution_score = _clamp_score(100 * min(width * height / 2_000_000, 1))

    contrast_raw = float(gray.std())
    contrast_score = _clamp_score(100 * contrast_raw / 60)

    column_gradient = np.mean(
        np.abs(np.diff(analysis_gray.astype(np.float32), axis=1)), axis=0
    )
    column_gradient -= column_gradient.mean()
    spectrum_1d = np.abs(np.fft.rfft(column_gradient))
    if len(spectrum_1d) >= 4:
        k0 = int(round(len(column_gradient) / 8))
        band = float(spectrum_1d[max(1, k0 - 1) : k0 + 2].mean())
        background = float(np.median(spectrum_1d[1:]))
        blockiness_raw = band / (background + 1e-6)
        blockiness_score = _clamp_score(100 / (1 + blockiness_raw / 10))
    else:
        blockiness_raw = None
        blockiness_score = None

    metrics = {
        "blur": _metric(blur_score, blur_raw, "Laplacian variance"),
        "skew": _metric(skew_score, skew_degrees, "degrees from horizontal/vertical"),
        "resolution": _metric(
            resolution_score, high_frequency_ratio, "high-frequency ratio"
        ),
        "contrast": _metric(contrast_score, contrast_raw, "grayscale standard deviation"),
        "blockiness": _metric(blockiness_score, blockiness_raw, "blockiness proxy ratio"),
    }

    available_scores = [
        item["score"] for item in metrics.values() if item["score"] is not None
    ]
    overall_score = _clamp_score(sum(available_scores) / len(available_scores))
    verdict = (
        "Quality looks good"
        if overall_score >= 75
        else "Review recommended"
        if overall_score >= 50
        else "Quality concerns detected"
    )

    explanations = {
        "blur": "Low sharpness proxy; consider checking focus or rescanning.",
        "skew": "Page lines appear tilted; consider aligning the scan.",
        "resolution": "Image pixel dimensions are limited; a larger scan may retain more detail.",
        "contrast": "Low grayscale contrast may make printed details harder to distinguish.",
        "blockiness": "Compression block artifacts may be present.",
    }
    issues = [
        explanations[name]
        for name, result in metrics.items()
        if result["score"] is not None and result["score"] < 75
    ]
    if metrics["skew"]["score"] is None:
        issues.append("Skew could not be estimated because too few long lines were detected.")
    if not issues:
        issues.append("No major issues were flagged by these quality proxy checks.")

    return {
        "width": width,
        "height": height,
        "metrics": metrics,
        "overall_score": overall_score,
        "verdict": verdict,
        "issues": issues,
        "notice": (
            "These are heuristic image-quality proxies, not calibrated clinical measures "
            "or a medical diagnosis."
        ),
    }


@app.get("/")
def home():
    filename = "index.html" if (ROOT / "index.html").is_file() else "medisense_quality_ui_demo.html"
    return send_from_directory(ROOT, filename)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/analyze")
def analyze_upload():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose an image file to analyze."}), 400

    image_bytes = uploaded.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        return jsonify({"error": "The selected file is empty."}), 400
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return jsonify({"error": "Image exceeds the 12 MB upload limit."}), 413

    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_image:
            if pil_image.format not in ALLOWED_FORMATS:
                return jsonify({"error": "Use a PNG, JPEG, TIFF, BMP, or WebP image."}), 400
            width, height = pil_image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                return jsonify({"error": "Image dimensions exceed the 20 megapixel limit."}), 413
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify({"error": "The uploaded file is not a valid supported image."}), 400

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "The image could not be decoded."}), 400

    return jsonify(analyze_image(image))


@app.errorhandler(RequestEntityTooLarge)
def handle_upload_too_large(error):
    return jsonify({"error": "Image exceeds the 12 MB upload limit."}), error.code or 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)