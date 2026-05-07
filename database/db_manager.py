"""
database/db_manager.py
Gestion SQLite : personnes, gabarits biométriques, événements d'authentification.
"""

import sqlite3
import numpy as np
import os

DB_PATH = "database/biometric_system.db"


class DatabaseManager:
    def __init__(self):
        os.makedirs("database", exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS persons (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT UNIQUE NOT NULL,
                status  TEXT NOT NULL        -- "authorized" | "unauthorized"
            );

            CREATE TABLE IF NOT EXISTS templates (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id  INTEGER NOT NULL,
                modality   TEXT NOT NULL,    -- "face" | "fingerprint"
                vector     BLOB NOT NULL,    -- vecteur numpy sérialisé
                checksum   TEXT NOT NULL,    -- SHA-256 du vecteur
                wm_payload TEXT NOT NULL,    -- payload du tatouage DCT
                FOREIGN KEY(person_id) REFERENCES persons(id)
            );

            CREATE TABLE IF NOT EXISTS auth_events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name  TEXT,
                decision     TEXT,           -- "authorized" | "unauthorized" | "unknown"
                face_score   REAL,
                fp_score     REAL,
                fused_score  REAL,
                detail       TEXT,
                timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    # ── Personnes ─────────────────────────────────────────────────────────────

    def add_person(self, name, status):
        """Insère ou récupère une personne. Retourne son id."""
        try:
            cur = self.conn.execute(
                "INSERT INTO persons (name, status) VALUES (?, ?)", (name, status))
            self.conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return self.get_person_id(name)

    def get_person_id(self, name):
        row = self.conn.execute(
            "SELECT id FROM persons WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

    def list_persons(self):
        return self.conn.execute("SELECT id, name, status FROM persons").fetchall()

    # ── Gabarits ──────────────────────────────────────────────────────────────

    def store_template(self, person_id, modality, vector, checksum, wm_payload):
        """Remplace le gabarit existant (une entrée par personne/modalité)."""
        blob = vector.astype(np.float32).tobytes()
        self.conn.execute(
            "DELETE FROM templates WHERE person_id=? AND modality=?",
            (person_id, modality))
        self.conn.execute(
            "INSERT INTO templates (person_id, modality, vector, checksum, wm_payload)"
            " VALUES (?,?,?,?,?)",
            (person_id, modality, blob, checksum, wm_payload))
        self.conn.commit()

    def load_template(self, person_id, modality):
        """Retourne le gabarit d'une personne ou None."""
        row = self.conn.execute(
            "SELECT vector, checksum, wm_payload FROM templates"
            " WHERE person_id=? AND modality=?",
            (person_id, modality)).fetchone()
        if row is None:
            return None
        vec = np.frombuffer(row[0], dtype=np.float32).copy()
        return {"vector": vec, "checksum": row[1], "wm_payload": row[2]}

    # ── Événements d'authentification ─────────────────────────────────────────

    def log_auth(self, person_name, decision, face_score,
                 fp_score, fused_score, detail):
        self.conn.execute(
            "INSERT INTO auth_events"
            " (person_name, decision, face_score, fp_score, fused_score, detail)"
            " VALUES (?,?,?,?,?,?)",
            (person_name, decision, face_score, fp_score, fused_score, detail))
        self.conn.commit()

    def get_auth_events(self, limit=50):
        return self.conn.execute(
            "SELECT * FROM auth_events ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()

    def close(self):
        self.conn.close()
