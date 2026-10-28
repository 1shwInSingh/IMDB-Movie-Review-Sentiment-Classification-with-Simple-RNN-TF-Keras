"""Automated Unit and Integration Tests for IMDB Sentiment Web Application."""
import json
import unittest

from app import app


class TestIMDBSentimentApp(unittest.TestCase):
    """Test suite covering routes, API endpoints, model inference, and error handling."""

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client with test mode enabled."""
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_index_page(self):
        """Verify the main UI page renders with status 200 and expected markup."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("CineSentiment", html)
        self.assertIn("Bi-LSTM", html)
        self.assertIn("Analyze Sentiment", html)

    def test_health_endpoint(self):
        """Verify /api/health returns healthy status and model information."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.get_data(as_text=True))
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("model"), "Bidirectional LSTM")
        self.assertTrue(data.get("weights_loaded"))
        self.assertGreater(data.get("vocab_size", 0), 0)

    def test_metrics_endpoint(self):
        """Verify /api/metrics returns the evaluation metrics from training."""
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.get_data(as_text=True))
        self.assertIn("test_accuracy", data)
        self.assertIn("test_loss", data)
        self.assertGreater(data["test_accuracy"], 0.70)

    def test_favicon_endpoint(self):
        """Verify /favicon.ico does not return a 404 error."""
        res = self.client.get("/favicon.ico")
        self.assertEqual(res.status_code, 204)

    def test_predict_positive_review(self):
        """Verify inference correctly classifies a clearly positive movie review."""
        payload = {
            "review": (
                "An absolute cinematic masterpiece! The performances were brilliant, "
                "the storytelling was captivating, and the direction was outstanding."
            )
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.get_data(as_text=True))

        self.assertEqual(data["label"], "POSITIVE")
        self.assertGreaterEqual(data["probability"], 0.5)
        self.assertGreater(data["confidence"], 50)
        self.assertIn("emoji", data)
        self.assertIn("verdict", data)
        self.assertIsInstance(data["latency_ms"], (int, float))

    def test_predict_negative_review(self):
        """Verify inference correctly classifies a clearly negative movie review."""
        payload = {
            "review": (
                "Utterly dreadful film. Terrible acting, incoherent script, "
                "and an absolute waste of two hours. Extremely boring."
            )
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.get_data(as_text=True))

        self.assertEqual(data["label"], "NEGATIVE")
        self.assertLess(data["probability"], 0.5)
        self.assertGreater(data["confidence"], 50)
        self.assertIn("emoji", data)
        self.assertIn("verdict", data)

    def test_predict_empty_review_returns_400(self):
        """Verify API rejects empty reviews with an informative 400 error."""
        res = self.client.post("/api/predict", json={"review": ""})
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.get_data(as_text=True))
        self.assertIn("error", data)

    def test_predict_whitespace_review_returns_400(self):
        """Verify API rejects reviews with only whitespace."""
        res = self.client.post("/api/predict", json={"review": "    \n\t  "})
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.get_data(as_text=True))
        self.assertIn("error", data)

    def test_predict_missing_payload_returns_400(self):
        """Verify API handles missing JSON payload gracefully."""
        res = self.client.post("/api/predict", data="not json", content_type="text/plain")
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.get_data(as_text=True))
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
