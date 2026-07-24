import os

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def connect():
    return psycopg.connect(
        host=os.environ["DB_HOST"],
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        row_factory=dict_row,
    )


def initialize_database():
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                email text PRIMARY KEY,
                resume_filename text NOT NULL,
                resume_content_type text NOT NULL,
                resume_data bytea NOT NULL,
                resume_text text,
                created_at timestamptz NOT NULL DEFAULT clock_timestamp()
            )
            """
        )
        connection.execute(
            """
            ALTER TABLE applications
            ADD COLUMN IF NOT EXISTS resume_text text
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interview_messages (
                id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                application_email text NOT NULL
                    REFERENCES applications(email) ON DELETE CASCADE,
                role text NOT NULL CHECK (role IN ('applicant', 'interviewer')),
                content text NOT NULL,
                generation_started_at timestamptz,
                generation_completed_at timestamptz,
                created_at timestamptz NOT NULL DEFAULT clock_timestamp()
            )
            """
        )
        connection.execute(
            """
            ALTER TABLE interview_messages
            ADD COLUMN IF NOT EXISTS generation_completed_at timestamptz
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS interview_messages_application_id
            ON interview_messages (application_email, id)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS belief_updates (
                id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                application_email text NOT NULL
                    REFERENCES applications(email) ON DELETE CASCADE,
                applicant_message_id bigint UNIQUE
                    REFERENCES interview_messages(id) ON DELETE CASCADE,
                posteriors jsonb NOT NULL,
                justifications jsonb NOT NULL,
                created_at timestamptz NOT NULL DEFAULT clock_timestamp()
            )
            """
        )
        connection.execute(
            """
            ALTER TABLE belief_updates
            ALTER COLUMN applicant_message_id DROP NOT NULL
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS belief_updates_application_id
            ON belief_updates (application_email, applicant_message_id)
            """
        )


def healthcheck():
    with connect() as connection:
        connection.execute("SELECT 1")


def create_application(
    email,
    resume_filename,
    resume_content_type,
    resume_data,
    resume_text,
):
    with connect() as connection:
        connection.execute(
            "DELETE FROM applications WHERE email = %s",
            (email,),
        )
        application = connection.execute(
            """
            INSERT INTO applications (
                email,
                resume_filename,
                resume_content_type,
                resume_data,
                resume_text
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING email, resume_filename, resume_content_type, created_at
            """,
            (
                email,
                resume_filename,
                resume_content_type,
                resume_data,
                resume_text,
            ),
        ).fetchone()
    return application


def add_message(
    application_email,
    role,
    content,
    generation_started_at=None,
    generation_completed_at=None,
):
    with connect() as connection:
        message = connection.execute(
            """
            INSERT INTO interview_messages (
                application_email,
                role,
                content,
                generation_started_at,
                generation_completed_at
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING
                id,
                role,
                content,
                generation_started_at,
                generation_completed_at,
                created_at
            """,
            (
                application_email,
                role,
                content,
                generation_started_at,
                generation_completed_at,
            ),
        ).fetchone()
    return message


def list_applications():
    with connect() as connection:
        return connection.execute(
            """
            SELECT
                a.email,
                a.resume_filename,
                a.created_at,
                count(m.id) AS message_count
            FROM applications a
            LEFT JOIN interview_messages m
                ON m.application_email = a.email
            GROUP BY a.email
            ORDER BY a.created_at DESC
            """
        ).fetchall()


def get_application(email):
    with connect() as connection:
        application = connection.execute(
            """
            SELECT email, resume_filename, resume_content_type, created_at
            FROM applications
            WHERE email = %s
            """,
            (email,),
        ).fetchone()
        messages = connection.execute(
            """
            SELECT
                id,
                role,
                content,
                generation_started_at,
                generation_completed_at,
                created_at
            FROM interview_messages
            WHERE application_email = %s
            ORDER BY id
            """,
            (email,),
        ).fetchall()
        belief_updates = connection.execute(
            """
            SELECT
                b.id,
                b.applicant_message_id,
                m.content AS applicant_message,
                b.posteriors,
                b.justifications,
                b.created_at
            FROM belief_updates b
            LEFT JOIN interview_messages m ON m.id = b.applicant_message_id
            WHERE b.application_email = %s
            ORDER BY b.id
            """,
            (email,),
        ).fetchall()
    return application, messages, belief_updates


def get_resume(email):
    with connect() as connection:
        return connection.execute(
            """
            SELECT resume_filename, resume_content_type, resume_data
            FROM applications
            WHERE email = %s
            """,
            (email,),
        ).fetchone()


def get_interview(email):
    with connect() as connection:
        resume = connection.execute(
            """
            SELECT resume_text
            FROM applications
            WHERE email = %s
            """,
            (email,),
        ).fetchone()
        messages = connection.execute(
            """
            SELECT role, content AS message
            FROM interview_messages
            WHERE application_email = %s
            ORDER BY id
            """,
            (email,),
        ).fetchall()
    return resume["resume_text"], messages


def get_resume_text(email):
    with connect() as connection:
        resume = connection.execute(
            """
            SELECT resume_text
            FROM applications
            WHERE email = %s
            """,
            (email,),
        ).fetchone()
    return resume["resume_text"]


def get_judge_context(email, applicant_message_id):
    with connect() as connection:
        resume = connection.execute(
            """
            SELECT resume_text
            FROM applications
            WHERE email = %s
            """,
            (email,),
        ).fetchone()
        messages = connection.execute(
            """
            SELECT role, content AS message
            FROM interview_messages
            WHERE application_email = %s AND id <= %s
            ORDER BY id
            """,
            (email, applicant_message_id),
        ).fetchall()
        previous = connection.execute(
            """
            SELECT posteriors, justifications
            FROM belief_updates
            WHERE application_email = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (email,),
        ).fetchone()
    if previous:
        previous = {
            dimension: {
                "posteriors": {
                    level: posterior[level]
                    for level in ("low", "medium", "high")
                },
                "justification": previous["justifications"][dimension],
            }
            for dimension, posterior in previous["posteriors"].items()
        }
    return resume["resume_text"], messages, previous


def add_belief_update(
    application_email,
    applicant_message_id,
    posteriors,
    justifications,
):
    with connect() as connection:
        return connection.execute(
            """
            INSERT INTO belief_updates (
                application_email,
                applicant_message_id,
                posteriors,
                justifications
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id, created_at
            """,
            (
                application_email,
                applicant_message_id,
                Jsonb(posteriors),
                Jsonb(justifications),
            ),
        ).fetchone()
