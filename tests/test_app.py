import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        self.data = app.DATA.copy()

    def test_dataset_has_required_columns(self):
        required = {
            "ticket_id",
            "fecha",
            "dia",
            "zona",
            "categoria",
            "prioridad",
            "canal",
            "tickets",
            "tiempo",
            "lat",
            "lon",
        }
        self.assertTrue(required.issubset(set(self.data.columns)))
        self.assertGreater(len(self.data), 0)

    def test_filters_reduce_or_equal_dataset(self):
        params = {
            "zona": "Kennedy",
            "prioridad": "Alta",
            "categoria": "Red",
            "canal": "Portal",
            "dia": "Lunes",
            "fecha_inicio": "",
            "fecha_fin": "",
        }
        filtered = app.apply_filters(self.data, params)
        self.assertLessEqual(len(filtered), len(self.data))
        if not filtered.empty:
            self.assertTrue((filtered["zona"] == "Kennedy").all())

    def test_summary_contains_core_metrics(self):
        summary = app.build_summary(self.data)
        for key in ["corr", "slope", "intercept", "avg_by_day", "zone_summary"]:
            self.assertIn(key, summary)

    def test_home_and_exports_status(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/export/filtered.csv").status_code, 200)
        self.assertEqual(self.client.get("/export/summary.xlsx").status_code, 200)
        self.assertEqual(self.client.get("/export/report.pdf").status_code, 200)


if __name__ == "__main__":
    unittest.main()
