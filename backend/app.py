import os
import sys
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

APP_ROOT = Path(__file__).resolve().parent
APP_ROOT_STR = str(APP_ROOT)

if APP_ROOT_STR in sys.path:
    sys.path.remove(APP_ROOT_STR)
sys.path.insert(0, APP_ROOT_STR)

if "services" in sys.modules:
    _mod = sys.modules["services"]
    _mod_file = getattr(_mod, "__file__", None)
    if not _mod_file or not str(_mod_file).startswith(APP_ROOT_STR + os.sep):
        del sys.modules["services"]

DB_PATH = os.environ.get("DB_PATH", "/data/app.db")
STORAGE_PATH = os.environ.get("STORAGE_PATH", "/data/storage")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(STORAGE_PATH, exist_ok=True)

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(
        app,
        resources={r"/*": {"origins": "*"}},
        allow_headers=["Content-Type", "Authorization", "X-User-Id"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )


    # Register APIs (keep app.py as the central register)
    from apis.system_api import bp as system_bp
    from apis.ingestion_api import bp as ingestion_bp
    from apis.csv_upload_api import bp as csv_upload_bp
    from apis.match_api import bp as match_bp
    from apis.auth_api import bp as auth_bp
    from apis.mdm_model_api import bp as mdm_model_bp
    from apis.source_input_api import bp as source_input_bp
    from apis.source_systems_api import bp as source_systems_bp
    from apis.matching_summary_api import bp as matching_summary_bp
    from apis.cleanup_recon_cluster_api import bp as cleanup_recon_cluster_bp
    from apis.cleanup_golden_record_api import bp as cleanup_golden_record_bp
    from apis.recon_cluster_records_api import bp as recon_cluster_records_bp
    from apis.golden_record_records_api import bp as golden_record_records_bp
    from apis.cluster_records_by_cluster_id_api import bp as records_by_cluster_id_bp


    app.register_blueprint(system_bp)
    app.register_blueprint(ingestion_bp, url_prefix="/api")
    app.register_blueprint(csv_upload_bp)
    app.register_blueprint(match_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(mdm_model_bp)
    app.register_blueprint(source_input_bp)
    app.register_blueprint(source_systems_bp)
    app.register_blueprint(matching_summary_bp)
    app.register_blueprint(cleanup_recon_cluster_bp)
    app.register_blueprint(cleanup_golden_record_bp)
    app.register_blueprint(recon_cluster_records_bp)
    app.register_blueprint(golden_record_records_bp)
    app.register_blueprint(records_by_cluster_id_bp)

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


