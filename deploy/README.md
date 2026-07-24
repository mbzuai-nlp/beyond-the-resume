# Local deployment

The stack contains an Nginx reverse proxy, separate applicant and reviewer frontends, a backend API, and PostgreSQL. The reviewer frontend is protected with HTTP Basic Auth and passes the authenticated username to the backend.

Copy `.env.example` to `.env`, then set the database credentials, reviewer credentials, and `OPENAI_API_KEY`:

```sh
docker compose up --build -d
```

- Applicant interview: `http://localhost/interview/`
- Reviewer console: `http://localhost/console/`

Replace `localhost` with the VM's address when accessing it over the local network. Stop the deployment with `docker compose down`. Database data remains in the `database_data` volume.

Inspect stored applications, résumés, and transcripts with:

```sh
docker compose exec database psql -U interviews -d interviews
```

The data is stored in the `applications`, `interview_messages`, and `belief_updates` tables.

The interview rubric is defined in `backend/rubric.json` and displayed on the reviewer console's Rubric screen.
Belief starts from the résumé, updates after applicant responses, and is shown as résumé/final snapshots plus per-response hover changes in the reviewer console.
