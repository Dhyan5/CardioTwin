"""Integration tests for the CardioTwin FastAPI endpoints (/api)."""

from fastapi.testclient import TestClient

from backend.api.main import app


def test_api_end_to_end_workflow():
    with TestClient(app) as client:
        # 1. Health & Schema
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "healthy"

        r_schema = client.get("/schema")
        assert r_schema.status_code == 200
        assert "heart_rate" in r_schema.json()["fields"]

        # 2. Model Evaluation Metrics
        r_metrics = client.get("/models/metrics")
        assert r_metrics.status_code == 200
        assert r_metrics.json()["auc_hybrid"] >= 0.85

        # 3. Patients & Twin Dashboard
        r_pts = client.get("/patients")
        assert r_pts.status_code == 200
        patients = r_pts.json()
        assert len(patients) >= 3

        r_twin = client.get("/twin/pt-001")
        assert r_twin.status_code == 200
        twin_data = r_twin.json()
        assert twin_data["patient_id"] == "pt-001"
        assert len(twin_data["trend"]) >= 12
        assert len(twin_data["prediction"]["top_factors"]) > 0

        # 4. Ingest new vitals -> increments state_version
        prev_ver = twin_data["twin_state"]["state_version"]
        r_ingest = client.post(
            "/twin/ingest",
            json={"patient_id": "pt-001", "source": "wearable_test", "heart_rate": 78.0, "systolic_bp": 134.0},
        )
        assert r_ingest.status_code == 200
        assert r_ingest.json()["twin_state"]["state_version"] == prev_ver + 1

        # 5. What-If Simulation comparing 3 plans over 90 days
        r_sim = client.post(
            "/whatif/simulate",
            json={
                "patient_id": "pt-001",
                "horizon_days": 90,
                "plans": [
                    {"plan_id": "no_tx", "label": "No Treatment", "medications": {}},
                    {
                        "plan_id": "current",
                        "label": "Current Plan",
                        "medications": {"metoprolol": 50, "lisinopril": 10, "furosemide": 20, "atorvastatin": 20, "aspirin": 81},
                    },
                    {
                        "plan_id": "proposed",
                        "label": "Optimized GDMT",
                        "medications": {"metoprolol": 100, "lisinopril": 20, "furosemide": 40, "atorvastatin": 40, "aspirin": 81},
                    },
                ],
            },
        )
        assert r_sim.status_code == 200
        sim_data = r_sim.json()
        assert len(sim_data["plans"]) == 3
        assert len(sim_data["plans"][0]["series"]) >= 13
        assert len(sim_data["delta_table"]) >= 5
