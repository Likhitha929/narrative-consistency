"""
app.py — interactive demo.

    pip install streamlit
    streamlit run app.py

Pick a character, type a claim (or load a real held-out test claim), and see
the evidence the retriever selected from the novel, the model's verdict, and
how the verdict changes when the evidence is removed.
"""

import html
import json
import re
from pathlib import Path

import streamlit as st

import config
import data_loader
from verify import BOOKS, Verifier

st.set_page_config(page_title="Narrative Consistency Checker", page_icon="📖", layout="centered")


@st.cache_resource(show_spinner="Loading model and book index…")
def load_verifier():
    return Verifier()


@st.cache_data
def characters():
    df = data_loader.load_train_csv(config.TRAIN_CSV)
    out = {}
    for b in BOOKS:
        sub = df[df.book_name.str.strip().str.lower() == b.lower()]
        out[b] = sub.char.value_counts().index.tolist()
    return out


@st.cache_data
def examples():
    p = Path("demo_examples.json")            # real claims from the held-out test split
    return json.loads(p.read_text()) if p.exists() else []


st.title("Narrative Consistency Checker")
st.caption("A 6M-parameter Transformer pretrained and fine-tuned from scratch on two novels. "
           "It selects evidence sentences from the book and judges whether a claim is consistent "
           "with the story or contradicts it.")

book = st.selectbox("Book", BOOKS)
ex_list = [e for e in examples() if e["book"].lower() == book.lower()]
ex_label = st.selectbox("Load a held-out test claim (optional)",
                        ["—"] + [f"{e['char']}: {e['claim'][:90]}" for e in ex_list])
ex = None if ex_label == "—" else ex_list[[f"{e['char']}: {e['claim'][:90]}" for e in ex_list].index(ex_label)]

chars = characters()[book]
character = st.selectbox("Character", chars, index=chars.index(ex["char"]) if ex and ex["char"] in chars else 0)
claim = st.text_area("Claim", value=ex["claim"] if ex else "", height=100)

if st.button("Check claim", type="primary", disabled=not claim.strip()):
    r = load_verifier().check(character, claim.strip(), book)
    p = r["p_consistent"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Verdict", r["verdict"].capitalize())
    c2.metric("P(consistent)", f"{p:.0%}")
    c3.metric("Without evidence", f"{r['p_without_evidence']:.0%}",
              delta=f"{(p - r['p_without_evidence']) * 100:+.0f} pts from evidence", delta_color="off")
    if ex and claim.strip() == ex["claim"]:
        ok = ex["label"] == r["verdict"]
        st.write(f"**Gold label:** {ex['label']} — model is {'✅ correct' if ok else '❌ wrong'}")

    st.subheader("Evidence selected from the book")
    ev = html.escape(r["evidence"]) or "<i>no evidence found</i>"
    for name in {w for w in re.findall(r"\w+", character) if len(w) > 2}:
        ev = re.sub(rf"\b({re.escape(name)})\b", r"<mark>\1</mark>", ev, flags=re.I)
    st.markdown(f"<div style='line-height:1.65'>{ev}</div>", unsafe_allow_html=True)

st.divider()
st.caption("Held-out accuracy on claims never seen in training is about 70% (a wording-only baseline "
           "gets 72%), so treat verdicts as indicative. Full evaluation in the README and report.")
