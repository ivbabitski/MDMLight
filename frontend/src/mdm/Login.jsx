"use client";

import React, { useEffect, useState } from "react";
import "./Login.css";
import { saveAuth } from "../authStorage";


export default function Login({ open, onClose, onLogin, onRegister, onRecoverEmail }) {

  const [view, setView] = useState("login"); // "login" | "register" | "recover"
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [statusKind, setStatusKind] = useState(""); // "ok" | "err"


  useEffect(() => {
    if (!open) return;

    function onKeyDown(e) {
      if (e.key === "Escape") {
        onClose?.();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  useEffect(() => {
    if (!open) return;
    setView("login");
    setUsername("");
    setPassword("");
    setEmail("");
    setBusy(false);
    setStatus("");
    setStatusKind("");
  }, [open]);


  if (!open) return null;

  function stop(e) {
    e.stopPropagation();
  }

  const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");

  async function apiPost(path, payload) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data?.error || `Request failed (${res.status})`);
    }
    return data;
  }


  async function submitLogin(e) {
    e.preventDefault();
    const payload = { username: username.trim(), password };

    setBusy(true);
    setStatus("");
    setStatusKind("");

    try {
      const out = await apiPost("/auth/login", payload);

      if (!out || !out.user_id || !out.username) {
        throw new Error("Auth API returned invalid payload (missing user_id/username)");
      }

      saveAuth({ user_id: out.user_id, username: out.username });

      const stored = window.localStorage.getItem("mdm_auth_v1");
      if (!stored) {
        throw new Error("Login succeeded but mdm_auth_v1 was not written (localStorage blocked or wrong build)");
      }

      window.localStorage.setItem("username", out.username);

      setStatusKind("ok");
      setStatus(`Login successful. user_id=${out.user_id}`);
      onLogin?.({ user_id: out.user_id, username: out.username });
      setTimeout(() => onClose?.(), 700);
    } catch (err) {
      setStatusKind("err");
      setStatus(err?.message ? String(err.message) : "Login failed.");
    } finally {
      setBusy(false);
    }
  }







  async function submitRegister(e) {
    e.preventDefault();
    const payload = { username: username.trim(), email: email.trim(), password };

    setBusy(true);
    setStatus("");
    setStatusKind("");

    try {
      await apiPost("/auth/register", payload);
      setStatusKind("ok");
      setStatus("Registered successfully.");
      onRegister?.({ username: payload.username, email: payload.email });
      setTimeout(() => onClose?.(), 900);
    } catch (err) {
      setStatusKind("err");
      setStatus(err?.message ? String(err.message) : "Registration failed.");
    } finally {
      setBusy(false);
    }
  }



  async function submitRecoverEmail(e) {
    e.preventDefault();
    const payload = { username: username.trim(), email: email.trim() };

    setBusy(true);
    setStatus("");
    setStatusKind("");

    try {
      await apiPost("/auth/recover", payload);
      setStatusKind("ok");
      setStatus("Recovery email sent.");
      onRecoverEmail?.({ username: payload.username });
      setTimeout(() => onClose?.(), 1100);
    } catch (err) {
      setStatusKind("err");
      setStatus(err?.message ? String(err.message) : "Recovery failed.");
    } finally {
      setBusy(false);
    }
  }



  const overlay = {
    position: "fixed",
    inset: 0,
    background: "rgba(0,0,0,0.45)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: 16,
    zIndex: 9999,
  };

  const modal = {
    width: "min(420px, 92vw)",
    background: "white",
    borderRadius: 14,
    boxShadow: "0 20px 60px rgba(0,0,0,0.25)",
    padding: 16,
  };

  const header = {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    marginBottom: 12,
  };

  const title = { margin: 0, fontSize: 18, fontWeight: 700 };

  const closeBtn = {
    border: "none",
    background: "transparent",
    cursor: "pointer",
    fontSize: 22,
    lineHeight: "22px",
    padding: "4px 8px",
  };

  const tabs = {
    display: "flex",
    gap: 8,
    marginBottom: 12,
  };

  const tabBtn = (active) => ({
    flex: 1,
    padding: "8px 10px",
    borderRadius: 10,
    border: active ? "1px solid var(--accent)" : "1px solid rgba(0,0,0,0.12)",
    background: active ? "var(--accentSoft2)" : "transparent",
    cursor: "pointer",
    fontWeight: 700,
  });


  const form = { display: "flex", flexDirection: "column", gap: 10 };

  const label = { display: "flex", flexDirection: "column", gap: 6, fontSize: 12, fontWeight: 700 };

  const input = {
    width: "100%",
    padding: "10px 12px",
    borderRadius: 10,
    border: "1px solid rgba(0,0,0,0.18)",
    outline: "none",
    fontSize: 14,
  };

  const primary = {
    width: "100%",
    padding: "10px 12px",
    borderRadius: 10,
    border: "1px solid rgba(0,0,0,0.22)",
     background: "var(--coral0)",
    color: "white",
    cursor: "pointer",
    fontWeight: 800,
    marginTop: 4,
  };

  const linkRow = { display: "flex", justifyContent: "space-between", gap: 8, marginTop: 6 };

  const link = {
    border: "none",
    background: "transparent",
    cursor: "pointer",
    padding: 0,
    textDecoration: "underline",
    fontSize: 12,
    fontWeight: 800,
  };

  return (
    <div
      className="login-overlay"
      style={overlay}
      onMouseDown={() => onClose?.()}
      role="dialog"
      aria-modal="true"
      aria-label="User login"
    >

      <div className="login-modal" style={modal} onMouseDown={stop}>
        <div style={header}>
          <h3 style={title}>User</h3>
          <button type="button" onClick={() => onClose?.()} style={closeBtn} aria-label="Close">
            ×
          </button>
        </div>

        <div style={tabs}>
          <button
            type="button"
            className="loginTab"
            data-active={view === "login"}
            style={tabBtn(view === "login")}
            onClick={() => setView("login")}
          >
            Login
          </button>

          <button
            type="button"
            className="loginTab"
            data-active={view === "register"}
            style={tabBtn(view === "register")}
            onClick={() => setView("register")}
          >
            New user
          </button>

          <button
            type="button"
            className="loginTab"
            data-active={view === "recover"}
            style={tabBtn(view === "recover")}
            onClick={() => setView("recover")}
          >
            Recover password
          </button>
        </div>


        {status ? (
          <div className={`loginStatus${statusKind ? ` loginStatus--${statusKind}` : ""}`}>
            {status}
          </div>
        ) : null}

        {view === "login" && (

          <form style={form} onSubmit={submitLogin}>
            <label style={label}>
              Username
              <input
                style={input}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
              />
            </label>

            <label style={label}>
              Password
              <input
                style={input}
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
              />
            </label>


            <button type="submit" className="loginPrimary" style={primary} disabled={busy}>
              Login
            </button>
          </form>
        )}

        {view === "register" && (
          <form style={form} onSubmit={submitRegister}>
            <label style={label}>
              Username
              <input
                style={input}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
              />
            </label>

            <label style={label}>
              Email
              <input
                style={input}
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
              />
            </label>

            <label style={label}>
              Password
              <input
                style={input}
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="new-password"
              />
            </label>

            <button type="submit" className="loginPrimary" style={primary} disabled={busy}>
              Create account
            </button>


            <div style={linkRow}>
              <button type="button" style={link} onClick={() => setView("login")}>
                Back to login
              </button>
              <span />
            </div>
          </form>
        )}

        {view === "recover" && (
          <form style={form} onSubmit={submitRecoverEmail}>
            <label style={label}>
              Username
              <input
                style={input}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
              />
            </label>

            <label style={label}>
              Email
              <input
                style={input}
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
              />
            </label>

            <button type="submit" className="loginPrimary" style={primary} disabled={busy}>
              Send recovery email
            </button>

            <div style={linkRow}>
              <button type="button" style={link} onClick={() => setView("login")}>
                Back to login
              </button>
              <span />
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
