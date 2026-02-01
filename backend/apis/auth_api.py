from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import os
import smtplib
import secrets
from email.message import EmailMessage

from db.sqlite_db import create_user, get_user_by_username, init_users, update_user_password


bp = Blueprint("auth", __name__, url_prefix="/auth")


def _json():
    return request.get_json(silent=True) or {}


def _send_email(to_email: str, subject: str, body: str) -> None:
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = os.environ.get("SMTP_PORT")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    smtp_from = os.environ.get("SMTP_FROM")

    if not smtp_host or not smtp_port or not smtp_user or not smtp_pass or not smtp_from:
        raise RuntimeError(
            "SMTP not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM."
        )

    msg = EmailMessage()
    msg["From"] = smtp_from
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


@bp.route("/register", methods=["POST"])
def register():
    init_users()
    data = _json()
    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip()
    password = str(data.get("password", ""))

    if not username or not email or not password:
        return jsonify({"ok": False, "error": "username, email, password are required"}), 400

    password_hash = generate_password_hash(password)

    try:
        create_user(username, email, password_hash)
    except Exception as e:
        msg = str(e).lower()
        if "unique" in msg or "constraint" in msg:
            return jsonify({"ok": False, "error": "username or email already exists"}), 409
        return jsonify({"ok": False, "error": "register failed"}), 500

    user_row = get_user_by_username(username)
    user_id = int(user_row["id"]) if user_row and user_row.get("id") is not None else None

    return jsonify({"ok": True, "user_id": user_id}), 200



@bp.route("/login", methods=["POST"])
def login():
    init_users()
    data = _json()
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))

    if not username or not password:
        return jsonify({"ok": False, "error": "username and password are required"}), 400

    user = get_user_by_username(username)
    if not user:
        return jsonify({"ok": False, "error": "invalid credentials"}), 401

    if not check_password_hash(user["password_hash"], password):
        return jsonify({"ok": False, "error": "invalid credentials"}), 401

    return jsonify({"ok": True, "user_id": int(user["id"]), "username": user["username"]}), 200



@bp.route("/recover", methods=["POST"])
def recover():
    init_users()
    data = _json()
    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip()

    if not username or not email:
        return jsonify({"ok": False, "error": "username and email are required"}), 400

    user = get_user_by_username(username)
    if not user:
        return jsonify({"ok": False, "error": "invalid credentials"}), 401

    if str(user["email"]).strip().lower() != email.lower():
        return jsonify({"ok": False, "error": "invalid credentials"}), 401

    temp_password = secrets.token_urlsafe(9)
    password_hash = generate_password_hash(temp_password)

    try:
        if not update_user_password(username, password_hash):
            return jsonify({"ok": False, "error": "recovery failed"}), 500

        _send_email(
            user["email"],
            "MDM Light password reset",
            f"Your temporary password is: {temp_password}\n\nLog in and change it.",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True}), 200

