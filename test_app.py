from __future__ import annotations

import io
import unittest

import cv2
import numpy as np

from app import app, analyze_image


class QualityApiTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_analysis_returns_all_metrics_for_uploaded_image(self):
        image = np.full((800, 1000, 3), 255, dtype=np.uint8)
        cv2.putText(image, "TEST SCAN", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)

        response = self.client.post(
            "/api/analyze",
            data={"image": (io.BytesIO(encoded.tobytes()), "scan.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual((body["width"], body["height"]), (1000, 800))
        self.assertEqual(
            set(body["metrics"]),
            {"blur", "skew", "resolution", "contrast", "blockiness"},
        )
        self.assertEqual(body["metrics"]["resolution"]["score"], 40)
        self.assertIn("heuristic", body["notice"])

    def test_rejects_non_image_upload(self):
        response = self.client.post(
            "/api/analyze",
            data={"image": (io.BytesIO(b"not an image"), "scan.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("valid supported image", response.get_json()["error"])

    def test_requires_an_image_field(self):
        response = self.client.post("/api/analyze", data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Choose an image", response.get_json()["error"])

    def test_constant_image_has_unavailable_skew(self):
        image = np.full((80, 80, 3), 255, dtype=np.uint8)
        result = analyze_image(image)
        self.assertIsNone(result["metrics"]["skew"]["score"])
        self.assertTrue(any("Skew could not be estimated" in item for item in result["issues"]))

    def test_detects_rotation_in_page_lines(self):
        image = np.full((800, 1000, 3), 255, dtype=np.uint8)
        for y in (180, 320, 460, 600):
            cv2.line(image, (50, y), (950, y), (0, 0, 0), 4)
        rotation = cv2.getRotationMatrix2D((500, 400), 6, 1.0)
        rotated = cv2.warpAffine(image, rotation, (1000, 800), borderValue=(255, 255, 255))

        result = analyze_image(rotated)

        self.assertAlmostEqual(result["metrics"]["skew"]["raw_value"], 6, delta=1)


if __name__ == "__main__":
    unittest.main()
