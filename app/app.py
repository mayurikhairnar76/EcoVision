import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils.config import WASTE_INFO, NAME_MR, TEXT, CONF_THRESHOLD
from utils.preprocessing import load_and_validate, to_model_input
from utils.prediction import load_artifacts, predict
from utils.gradcam import make_gradcam, overlay_heatmap

st.set_page_config(page_title="EcoVision", page_icon="♻️", layout="wide")
st.markdown("""<style>
.hero{background:linear-gradient(90deg,#1b5e20,#43a047);padding:1.2rem 1.5rem;border-radius:14px;color:#fff;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem}.hero p{margin:0;opacity:.9}
.card{background:#f1f8e9;border-left:6px solid #43a047;padding:.8rem 1rem;border-radius:10px;margin:.4rem 0;color:#1b2b1b}
</style>""", unsafe_allow_html=True)

ss = st.session_state
for k, v in dict(count=0, last="-", rec=0, nonrec=0, uk=0, result=None).items():
    ss.setdefault(k, v)

lang = st.sidebar.radio("Language / भाषा", ["English", "मराठी"], horizontal=True)
t, mr = TEXT[lang], lang == "मराठी"

@st.cache_resource
def get_model():
    return load_artifacts()

st.markdown(f"<div class='hero'><h1>♻️ {t['title']}</h1><p>{t['sub']}</p></div>", unsafe_allow_html=True)

try:
    model, classes, metrics = get_model()
except RuntimeError as e:
    st.error(str(e)); st.stop()

# ---- Dashboard (accuracy is read from model/metrics.json written by the training notebook) ----
st.subheader(t["dash"])
c = st.columns(5)
acc = metrics.get("test_accuracy")
last = "-" if ss.last == "-" else (NAME_MR[ss.last] if mr else ss.last.title())
c[0].metric(t["cats"], len(classes))
c[1].metric(t["acc"], f"{acc*100:.2f}%" if acc is not None else "N/A")
c[2].metric(t["preds"], ss.count)
c[3].metric(t["last"], last)
c[4].metric(t["rec_cnt"], f"{ss.rec} / {ss.nonrec}")
st.divider()

# ---- Upload / preview / predict / reset ----
left, right = st.columns(2)
img = None
with left:
    file = st.file_uploader(t["upload"], key=f"up{ss.uk}")   # no type filter: we show our own format error
    if file is not None:
        try:
            img = load_and_validate(file)
            st.image(img, caption=t["orig"], use_container_width=True)
        except ValueError as e:
            st.error(str(e))
    b1, b2 = st.columns(2)
    go = b1.button(t["predict"], type="primary", use_container_width=True)
    if b2.button(t["reset"], use_container_width=True):
        ss.uk += 1; ss.result = None; st.rerun()

if go:
    if file is None:
        st.warning(t["noimg"])
    elif img is not None:
        batch = to_model_input(img)
        name, idx, conf, probs = predict(model, classes, batch)
        cam = make_gradcam(model, batch, idx)
        ss.result = dict(name=name, conf=conf, probs=probs, heat=overlay_heatmap(img, cam), orig=img)
        ss.count += 1
        if conf >= CONF_THRESHOLD:
            ss.last = name
            ss.rec += int(WASTE_INFO[name]["recyclable"]); ss.nonrec += int(not WASTE_INFO[name]["recyclable"])
        st.rerun()

# ---- Result ----
r = ss.result
if r and file is not None:
    with right:
        if r["conf"] < CONF_THRESHOLD:
            st.warning(t["low"]); st.caption(f"{t['conf']}: {r['conf']*100:.1f}%")
        else:
            w = WASTE_INFO[r["name"]]
            nm = f"{NAME_MR[r['name']]} ({r['name'].title()})" if mr else r["name"].title()
            st.markdown(f"<div class='card'><b>{t['waste']}:</b> {nm}<br><b>{t['conf']}:</b> {r['conf']*100:.1f}%<br>"
                        f"<b>{t['bin']}:</b> {w['bin_mr'] if mr else w['bin']}</div>", unsafe_allow_html=True)
            st.progress(int(r["conf"] * 100))
            m = st.columns(3)
            m[0].metric(t["eco"], f"{w['eco']}/100")
            m[1].metric(t["recyclable"], t["yes"] if w["recyclable"] else t["no"])
            m[2].metric(t["impact"], w["impact_mr"] if mr else w["impact"])
            st.info(f"**{t['info']}:** {w['info_mr'] if mr else w['info']}")
            st.success(f"**{t['tip']}:** {w['tip_mr'] if mr else w['tip']}")
            st.caption(t["disclaimer"])
        with st.expander("All class probabilities"):
            st.bar_chart({k: float(v) for k, v in zip(classes, r["probs"])})

    if r["conf"] >= CONF_THRESHOLD:
        st.divider(); st.subheader(t["xai"])
        x1, x2 = st.columns(2)
        x1.image(r["orig"], caption=t["orig"], use_container_width=True)
        x2.image(r["heat"], caption=t["heat"], use_container_width=True)
        st.caption(t["xai_note"])
