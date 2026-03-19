from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from src.ai_pipeline import VoicePipeline
from src.rag import SlideRetriever, resolve_env

# Load committed non-secret defaults first, then local developer overrides.
load_dotenv(".env.shared")
load_dotenv()

# In Streamlit Community Cloud, secrets are set in App settings -> Secrets.
# Mirror known keys into environment variables for existing code paths.
try:
    secrets_map = dict(st.secrets)
except Exception:
    secrets_map = {}

for key in [
    "CHAT_MODEL",
    "WHISPER_MODEL",
    "TTS_MODEL",
    "TTS_VOICE",
    "CHROMA_PERSIST_DIRECTORY",
    "CHROMA_COLLECTION_NAME",
    "EMBEDDING_MODEL",
    "TOP_K",
    "MIN_RELEVANCE_SCORE",
    "COURSE_WEEK_FILTER",
]:
    if not os.getenv(key) and key in secrets_map:
        os.environ[key] = str(secrets_map[key])

st.set_page_config(page_title="Analytics and Society Voice Copilot", page_icon="🎙️", layout="wide")


@st.cache_resource
def get_services() -> Dict[str, Any]:
    env = resolve_env()
    retriever = SlideRetriever(
        persist_directory=env["persist_directory"],
        collection_name=env["collection_name"],
        embedding_model=env["embedding_model"],
        top_k=env["top_k"],
        min_relevance_score=env["min_relevance_score"],
    )
    pipeline = VoicePipeline()
    return {"retriever": retriever, "pipeline": pipeline, "env": env}


def confidence_label(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


MODE_SUGGESTIONS: Dict[str, List[str]] = {
    "Classroom Tutor": [
        "Explain the ABACUS and ROBOTS frameworks in simple terms for this course.",
        "How should we evaluate fairness trade-offs in education AI tools?",
        "What are the strongest arguments for and against AI proctoring?",
    ],
    "Toolkit Builder": [
        "Create an ABACUS/ROBOTS checklist for AI tutoring systems in universities.",
        "Draft mitigation questions for bias and surveillance in EdTech analytics.",
        "Generate practical assessment prompts education leaders can use before deploying an AI tool.",
    ],
    "Case Review": [
        "Review a case where a university uses predictive analytics for dropout risk and recommend safeguards.",
        "Analyze an AI plagiarism detector in education and provide a go/no-go recommendation.",
        "Evaluate adaptive learning systems from students, teachers, and institution perspectives.",
    ],
}


def build_artifact(turn: Dict[str, Any]) -> str:
    refs = []
    for i, chunk in enumerate(turn.get("chunks", []), start=1):
        meta = chunk.metadata
        lecture = meta.get("lecture_title", "Unknown lecture")
        slide = meta.get("slide_number", "?")
        refs.append(f"- Doc {i}: {lecture}, Slide {slide}, score={chunk.score:.2f}")

    ref_block = "\n".join(refs) if refs else "- No retrieved references"
    return (
        f"Mode: {turn.get('mode', 'Classroom Tutor')}\n"
        f"Question: {turn.get('question', '')}\n\n"
        f"Response:\n{turn.get('answer', '')}\n\n"
        f"Retrieved references:\n{ref_block}\n"
    )


st.title("Analytics and Society Voice Copilot")
st.caption(
    "Chat with your course materials using text or voice. The app transcribes with Whisper, "
    "retrieves from your slides (LangChain + Chroma), generates grounded answers, and can autoplay audio replies."
)

services = get_services()
retriever = services["retriever"]
pipeline = services["pipeline"]
default_week_filter = services["env"]["course_week_filter"]

with st.sidebar:
    st.subheader("Session controls")
    mode = st.selectbox(
        "Mode",
        options=["Classroom Tutor", "Toolkit Builder", "Case Review"],
        index=1,
        help="Choose how the assistant structures answers for your education/edtech project.",
    )
    autoplay = st.toggle("Autoplay answer audio", value=True)
    week_filter = st.text_input("Optional course week filter", value=default_week_filter or "")
    show_context = st.toggle("Show retrieved context", value=True)
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state["chat_history"] = []
        st.session_state["last_audio_hash"] = ""
        st.rerun()
    st.divider()
    st.markdown("Suggested prompts")
    suggestions = MODE_SUGGESTIONS[mode]
    for text in suggestions:
        if st.button(text, use_container_width=True):
            st.session_state["typed_prompt"] = text

st.info(
    "This tool is tailored to your course project: an education/edtech AI ethics toolkit based on "
    "ABACUS/ROBOTS and practical recommendations."
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "last_audio_hash" not in st.session_state:
    st.session_state["last_audio_hash"] = ""


def process_question(user_question: str, source_label: str) -> None:
    with st.status("Processing", expanded=True) as status:
        status.write("Retrieving slide context...")
        chunks = retriever.search(user_question, week_filter.strip() or None)

        status.write("Generating answer...")
        answer_text, confidence = pipeline.generate_answer(user_question, chunks, mode)

        status.write("Synthesizing speech...")
        audio_bytes = pipeline.synthesize_speech(answer_text)

        status.update(label="Done", state="complete")

    st.session_state["chat_history"].append(
        {
            "question": user_question,
            "answer": answer_text,
            "confidence": confidence,
            "chunks": chunks,
            "audio": audio_bytes,
            "source": source_label,
            "mode": mode,
        }
    )


st.markdown("### Conversation")
for idx, turn in enumerate(st.session_state["chat_history"]):
    with st.chat_message("user"):
        st.write(turn["question"])
        st.caption(f"Input: {turn.get('source', 'text')}")

    with st.chat_message("assistant"):
        st.write(turn["answer"])
        st.caption(
            f"Mode: {turn.get('mode', 'Classroom Tutor')} | "
            f"Grounding confidence: {confidence_label(turn['confidence'])} ({turn['confidence']:.2f})"
        )
        is_latest = idx == len(st.session_state["chat_history"]) - 1
        st.audio(turn["audio"], format="audio/mp3", autoplay=(autoplay and is_latest))

        if turn.get("mode") in {"Toolkit Builder", "Case Review"}:
            artifact_text = build_artifact(turn)
            st.download_button(
                label="Download this output as project artifact",
                data=artifact_text,
                file_name=f"artifact_turn_{idx + 1}.txt",
                mime="text/plain",
                key=f"artifact_download_{idx}",
            )

        if show_context:
            with st.expander("Retrieved evidence"):
                if not turn["chunks"]:
                    st.warning(
                        "No chunks passed the relevance threshold. Try a more specific question "
                        "or remove the week filter."
                    )
                else:
                    for i, chunk in enumerate(turn["chunks"], start=1):
                        meta = chunk.metadata
                        lecture = meta.get("lecture_title", "Unknown lecture")
                        slide = meta.get("slide_number", "?")
                        st.markdown(
                            f"**Doc {i}** | {lecture} | Slide {slide} | score={chunk.score:.2f}"
                        )
                        st.write(chunk.content)

# Text-first composer
prompt_override = st.session_state.pop("typed_prompt", "")
if prompt_override:
    st.info(f"Suggested prompt selected: {prompt_override}")

typed_question = st.chat_input(f"[{mode}] Ask about Analytics and Society content...")
if prompt_override and not typed_question:
    typed_question = prompt_override

if typed_question and typed_question.strip():
    process_question(typed_question.strip(), "text")
    st.rerun()

# Voice input option under chat composer
st.markdown("#### Voice input")
audio_file = st.audio_input("Record and send a voice question")
send_voice = st.button("Send voice message", type="secondary", use_container_width=True)

if send_voice:
    if audio_file is None:
        st.warning("Record audio first, then click 'Send voice message'.")
    else:
        audio_bytes = audio_file.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if audio_hash == st.session_state["last_audio_hash"]:
            st.info("Same recording already sent. Record a new message to send again.")
        else:
            st.session_state["last_audio_hash"] = audio_hash
            with st.status("Processing", expanded=True) as status:
                status.write("Transcribing audio...")
                user_question = pipeline.transcribe_audio_bytes(audio_bytes)
                status.write(f"Transcription: {user_question}")
                status.update(label="Transcription complete", state="complete")

            process_question(user_question, "voice")
            st.rerun()
