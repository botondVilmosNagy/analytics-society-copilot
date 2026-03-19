# Analytics and Society Voice Copilot (MVP)

Audio-to-audio chatbot for the Analytics and Society course:

1. User speaks in the web UI.
2. OpenAI Whisper transcribes speech to text.
3. Transcript queries existing LangChain + Chroma vector database.
4. Retrieved slide context is passed to an LLM.
5. Answer is converted to speech and auto-played.

## Requirements

- Python 3.10+
- Existing Chroma DB already built from course slides
- OpenAI API key with access to Whisper, chat, and TTS models

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Update `.env` values to match your roommate's vector DB:

- `CHROMA_PERSIST_DIRECTORY`
- `CHROMA_COLLECTION_NAME`
- `EMBEDDING_MODEL` (must match ingestion setup)

## Where to put the large PDF

Place the combined course slide PDF here:

`data/slides/analytics_and_society_slides.pdf`

If you use a different path or filename, set `SLIDES_PDF_PATH` in `.env`.

## Initialize or rebuild the vector database from PDF

If you want to build/update Chroma directly from the large PDF in this project:

```bash
python scripts/ingest_slides.py
```

If you already ingested once and want to fix metadata (week/lecture) cleanly, rebuild with reset:

```bash
python scripts/ingest_slides.py --reset-collection
```

Optional custom path:

```bash
python scripts/ingest_slides.py --pdf-path "data/slides/your_file.pdf"
```

The script:

1. Reads the PDF page by page.
2. Stores each page as slide metadata (`slide_number`).
3. Infers and stores `course_week` and `lecture_title` (carried forward across slides).
3. Splits text into chunks.
4. Embeds with OpenAI embeddings.
5. Writes to `CHROMA_COLLECTION_NAME` in `CHROMA_PERSIST_DIRECTORY`.

Run app:

```bash
streamlit run app.py
```

## After publishing to GitHub

### 1) Configure repository secrets

Go to your repository settings:

`Settings -> Secrets and variables -> Actions -> New repository secret`

Add at least:

- `OPENAI_API_KEY`

Optional (if you later move environment values into Actions workflows):

- `CHAT_MODEL`
- `WHISPER_MODEL`
- `TTS_MODEL`
- `TTS_VOICE`

Note: do **not** commit `.env`. This repo keeps `.env` ignored and uses `.env.example` for template values.

### 2) CI validation

This repo includes a GitHub Actions workflow at `.github/workflows/ci.yml` that runs on push and pull requests.

It checks:

1. Dependency installation
2. Python syntax compilation (`compileall`)
3. Import smoke test for core modules
4. Optional OpenAI client initialization when `OPENAI_API_KEY` secret exists

## Streamlit Community Cloud deployment

This repository is prepared so deployment has all required resources:

- Slide PDF: `data/slides/context.pdf`
- Prebuilt vector DB: `context_db/`

### Steps

1. Open Streamlit Community Cloud and click `New app`.
2. Select repository: `botondVilmosNagy/analytics-society-copilot`.
3. Branch: `main`.
4. Main file path: `app.py`.
5. In `Advanced settings` -> `Secrets`, add:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```

Optional extra keys can be copied from `.streamlit/secrets.toml.example`.

6. Deploy.

### Notes

- The app reads from `.env` locally and from Streamlit Secrets in cloud.
- Do not commit real secrets to git.
- If you later regenerate the vector DB locally, commit updated `context_db/` files before redeploying.

## UX notes

- Chat-style message bubbles (`st.chat_message`)
- Text chat composer (`st.chat_input`)
- Voice message option (`st.audio_input` + send button)
- Three syllabus-aligned modes:
	- `Classroom Tutor` for concept understanding
	- `Toolkit Builder` for ABACUS/ROBOTS toolkit creation
	- `Case Review` for decision-oriented case analysis
- Visible processing states (transcribe -> retrieve -> generate -> speak)
- Optional week filter for retrieval
- Autoplay for newest assistant audio response
- Expandable evidence panel with slide references and retrieval scores
- Downloadable text artifact per response in Toolkit/Case modes (useful for group project documentation)

## Retrieval troubleshooting

- Keep `COURSE_WEEK_FILTER` empty unless your collection actually has `course_week` metadata.
- If retrieval seems empty, set `MIN_RELEVANCE_SCORE=0.0` first.
- Confirm your `.env` points to the right collection and persist directory.

## Next implementation tasks

1. Add lecture picker based on metadata values from the DB.
2. Add reranking stage for long-context precision.
3. Add session export for study notes.
4. Add guardrails for explicit out-of-scope requests.
