import time
import datetime
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "gbr_model.pkl"
SCALER_PATH = "scaler.pkl"

APP_TITLE = "Employee Salary Predictor"
APP_ICON = "💼"
DEVELOPER = "Naved Alam"
VERSION = "3.0"

GENDER_MAP = {"Female": 0, "Male": 1}
EDUCATION_MAP = {"Diploma": 0, "Bachelor": 1, "PhD": 2, "Master": 3}
DEPARTMENT_MAP = {"Operations": 0, "IT": 1, "Sales": 2, "HR": 3, "Marketing": 4}
LEVEL_MAP = {"Junior": 0, "Manager": 1, "Mid": 2, "Senior": 3, "Lead": 4}
REMOTE_MAP = {"No": 0, "Yes": 1}
CITY_MAP = {"Hyderabad": 0, "Mumbai": 1, "Chennai": 2, "Delhi": 3}

SCALE_COLS = [
    "Experience_Years", "Performance_Rating", "Certifications",
    "Overtime_Hours", "Company_Tenure", "Projects_Completed", "Skill_Score",
]

FEATURE_ORDER = [
    "Age", "Gender", "Education", "Experience_Years", "Department",
    "Job_Level", "Performance_Rating", "Certifications", "Overtime_Hours",
    "Remote_Work", "City", "Company_Tenure", "Projects_Completed", "Skill_Score",
]

# Approximate salary range (LPA) used to scale the gauge chart.
# Adjust to match the real distribution of your training data.
SALARY_MIN = 2.0
SALARY_MAX = 60.0

DEFAULTS = {
    "age": 25, "gender": "Female", "education": "Bachelor", "city": "Hyderabad",
    "dept": "IT", "level": "Junior", "remote": "No", "exp": 2.0, "tenure": 2.0,
    "cert": 0, "over": 10, "proj": 5, "perf": 3, "skill": 50,
}

DARK_PAPER = "rgba(0,0,0,0)"
DARK_FONT = "#e5e7eb"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# =========================================================
# STYLES
# =========================================================
st.markdown("""
<style>

/* ============================================================
   ANIMATED GRADIENT BACKGROUND (dark theme)
============================================================ */
.stApp{
    background:linear-gradient(120deg,#05070f,#0b1330,#132a5e,#0b1330,#05070f);
    background-size:400% 400%;
    animation:gradientBG 18s ease infinite;
}
@keyframes gradientBG{
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}
.stApp::before{
    content:"";
    position:fixed;
    inset:0;
    background:
        radial-gradient(circle at 15% 20%, rgba(59,130,246,.12), transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(99,102,241,.12), transparent 40%);
    pointer-events:none;
    z-index:0;
}

/* ============================================================
   REMOVE STREAMLIT DEFAULTS
============================================================ */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

/* ============================================================
   MAIN CONTAINER / RESPONSIVE PADDING
============================================================ */
.block-container{
    padding-top:1.4rem;
    padding-left:3rem;
    padding-right:3rem;
    padding-bottom:2rem;
    max-width:1300px;
}
@media (max-width:768px){
    .block-container{ padding-left:1rem; padding-right:1rem; }
    .hero-content h1{ font-size:32px !important; }
    .hero-pill{ font-size:12px !important; padding:5px 10px !important; }
    .money{ font-size:40px !important; }
}

/* ============================================================
   GLASSMORPHISM CARD (base) — used for pure static HTML blocks
============================================================ */
.glass{
    background:rgba(255,255,255,0.055);
    border:1px solid rgba(255,255,255,0.12);
    border-radius:22px;
    padding:26px 28px;
    backdrop-filter:blur(14px);
    -webkit-backdrop-filter:blur(14px);
    box-shadow:0 10px 40px rgba(0,0,0,.30);
    margin-bottom:24px;
}

/* ============================================================
   GLASSMORPHISM CARD for st.container(border=True) — this is the
   one that actually wraps real widgets (inputs, charts, forms)
============================================================ */
[data-testid="stVerticalBlockBorderWrapper"]{
    background:rgba(255,255,255,0.055) !important;
    border:1px solid rgba(255,255,255,0.12) !important;
    border-radius:22px !important;
    backdrop-filter:blur(14px);
    -webkit-backdrop-filter:blur(14px);
    box-shadow:0 10px 40px rgba(0,0,0,.30);
    margin-bottom:20px;
    padding:6px;
}

/* ============================================================
   HERO
============================================================ */
.hero{
    width:100%;
    padding:44px 20px;
    display:flex;
    justify-content:center;
    align-items:center;
    border-radius:28px;
    background:linear-gradient(135deg, rgba(37,99,235,.55), rgba(29,78,216,.55), rgba(30,58,138,.65));
    border:1px solid rgba(255,255,255,0.15);
    backdrop-filter:blur(10px);
    box-shadow:0 20px 60px rgba(0,0,0,.4);
    margin-bottom:28px;
    overflow:hidden;
    position:relative;
}
.hero::before{
    content:"";
    position:absolute;
    top:-60%; left:-60%;
    width:220%; height:220%;
    background:radial-gradient(circle, rgba(255,255,255,0.07) 0%, transparent 60%);
    animation:rotateGlow 24s linear infinite;
}
@keyframes rotateGlow{ from{transform:rotate(0deg);} to{transform:rotate(360deg);} }
.hero-content{ text-align:center; position:relative; z-index:1; }
.hero-content h1{
    font-size:50px; font-weight:900; margin:0 0 8px 0; letter-spacing:-1px;
    background:linear-gradient(90deg,#ffffff,#bfdbfe,#93c5fd,#ffffff);
    background-size:300% auto;
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
    animation:shimmerText 6s linear infinite;
}
@keyframes shimmerText{
    0%{ background-position:0% 50%; }
    100%{ background-position:300% 50%; }
}

.hero-subtitle{
    display:flex;
    align-items:center;
    justify-content:center;
    flex-wrap:wrap;
    gap:10px;
    margin-top:6px;
    animation:fadeInUp .6s ease .1s both;
}
.hero-pill{
    display:inline-flex;
    align-items:center;
    gap:6px;
    padding:8px 18px;
    border-radius:999px;
    background:rgba(255,255,255,0.12);
    border:1px solid rgba(255,255,255,0.22);
    color:#eff6ff;
    font-size:14.5px;
    font-weight:600;
    letter-spacing:.2px;
    backdrop-filter:blur(6px);
    transition:.25s ease;
}
.hero-pill:hover{
    background:rgba(255,255,255,0.2);
    transform:translateY(-2px);
}
.hero-sep{
    color:rgba(255,255,255,0.5);
    font-size:16px;
}
@keyframes fadeInUp{
    0%{ opacity:0; transform:translateY(8px); }
    100%{ opacity:1; transform:translateY(0); }
}
@media (max-width:768px){
    .hero-pill{ font-size:12.5px; padding:6px 12px; }
}
.money{ font-size:54px; margin-bottom:8px; animation:float 2.4s ease-in-out infinite; display:inline-block; }
@keyframes float{ 0%{transform:translateY(0px);} 50%{transform:translateY(-10px);} 100%{transform:translateY(0px);} }

/* ============================================================
   NAV PILLS (sidebar radio)
============================================================ */
div[role="radiogroup"] > label{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.10);
    border-radius:14px;
    padding:10px 14px !important;
    margin-bottom:6px;
    transition:.2s ease;
}
div[role="radiogroup"] > label:hover{
    background:rgba(59,130,246,0.18);
    border-color:rgba(96,165,250,0.5);
}

/* ============================================================
   METRICS
============================================================ */
[data-testid="stMetric"]{
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.12);
    border-radius:16px;
    padding:14px 10px;
    backdrop-filter:blur(6px);
}
[data-testid="stMetricLabel"]{ color:#93c5fd !important; }
[data-testid="stMetricValue"]{ color:white !important; }

/* ============================================================
   SECTION TITLES
============================================================ */
.section-title{
    font-size:21px; font-weight:700; color:white;
    margin-top:6px; margin-bottom:16px;
    display:flex; align-items:center; gap:8px;
    border-left:4px solid #3b82f6; padding-left:10px;
}

/* ============================================================
   TEXT DEFAULTS
============================================================ */
label, .stMarkdown, .stCaption, p, span, li{ color:#e5e7eb; }
h1,h2,h3,h4{ color:#ffffff; }

/* ============================================================
   INPUTS
============================================================ */
input, textarea{ border-radius:12px !important; }
div[data-baseweb="select"] > div{
    background-color:rgba(255,255,255,0.08) !important;
    border-radius:12px !important;
    color:white !important;
    border:1px solid rgba(255,255,255,0.15) !important;
}
.stNumberInput input, .stTextInput input, .stTextArea textarea{
    background-color:rgba(255,255,255,0.08) !important;
    color:white !important;
    border:1px solid rgba(255,255,255,0.15) !important;
}
.stSlider{ padding-top:10px; padding-bottom:6px; }

/* ============================================================
   BUTTONS
============================================================ */
.stButton>button{
    width:100%; height:58px; border-radius:16px;
    font-size:20px; font-weight:700; border:none; color:white;
    background:linear-gradient(90deg,#2563eb,#4f46e5);
    transition:.3s ease;
    box-shadow:0 8px 24px rgba(37,99,235,.35);
}
.stButton>button:hover{
    transform:translateY(-3px);
    box-shadow:0 15px 40px rgba(37,99,235,.5);
}
button[kind="secondary"]{
    background:linear-gradient(90deg,#6b7280,#4b5563) !important;
    box-shadow:0 8px 24px rgba(75,85,99,.35) !important;
    color:white !important;
}

/* ============================================================
   RESULT CARD
============================================================ */
.result-card{
    padding:30px; border-radius:22px;
    background:linear-gradient(120deg,#16a34a,#22c55e,#15803d);
    background-size:200% 200%;
    animation:shine 6s ease infinite, popIn .5s ease;
    color:white; text-align:center; margin-top:10px;
    box-shadow:0 15px 45px rgba(34,197,94,.4);
}
@keyframes shine{ 0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;} }
@keyframes popIn{ 0%{transform:scale(0.92);opacity:0;} 100%{transform:scale(1);opacity:1;} }
.result-card .label{ font-size:16px; opacity:.9; font-weight:500; }
.result-card .amount{ font-size:42px; font-weight:900; margin:6px 0; }
.result-card .sub{ font-size:14px; opacity:.85; }

/* ============================================================
   SUMMARY PANEL
============================================================ */
.summary-item{
    display:flex; justify-content:space-between;
    padding:8px 4px; border-bottom:1px dashed rgba(255,255,255,0.12);
    font-size:14px;
}
.summary-item span:first-child{ color:#93c5fd; }
.summary-item span:last-child{ color:white; font-weight:600; }

/* ============================================================
   BADGES
============================================================ */
.badge{
    display:inline-block; padding:5px 14px; border-radius:999px;
    background:rgba(59,130,246,0.18); border:1px solid rgba(96,165,250,0.4);
    color:#bfdbfe; font-size:13px; font-weight:600; margin:3px 4px 3px 0;
}

/* ============================================================
   LOADING ANIMATION
============================================================ */
.loader-wrap{
    display:flex; align-items:center; justify-content:center; gap:10px;
    padding:14px; color:#93c5fd; font-weight:600;
}
.loader-dot{ width:10px; height:10px; border-radius:50%; background:#3b82f6; animation:bounce 1s infinite ease-in-out; }
.loader-dot:nth-child(2){ animation-delay:.15s; }
.loader-dot:nth-child(3){ animation-delay:.3s; }
@keyframes bounce{ 0%,80%,100%{transform:scale(0.6); opacity:.5;} 40%{transform:scale(1); opacity:1;} }

/* ============================================================
   LIVE ESTIMATE CHIP
============================================================ */
.live-chip{
    display:flex;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:10px;
    padding:16px 22px;
    border-radius:18px;
    background:linear-gradient(90deg, rgba(59,130,246,0.16), rgba(99,102,241,0.16));
    border:1px solid rgba(96,165,250,0.35);
    margin-bottom:18px;
}
.live-chip .live-label{
    display:flex; align-items:center; gap:8px;
    color:#bfdbfe; font-weight:600; font-size:14px;
}
.live-dot{
    width:9px; height:9px; border-radius:50%;
    background:#22c55e;
    box-shadow:0 0 0 rgba(34,197,94,.6);
    animation:pulseDot 1.6s infinite;
}
@keyframes pulseDot{
    0%{ box-shadow:0 0 0 0 rgba(34,197,94,.6); }
    70%{ box-shadow:0 0 0 8px rgba(34,197,94,0); }
    100%{ box-shadow:0 0 0 0 rgba(34,197,94,0); }
}
.live-chip .live-amount{
    color:white; font-size:26px; font-weight:800;
}

/* ============================================================
   INFLUENCE / PERCENTILE LABELS
============================================================ */
.influence-note{
    color:#93c5fd; font-size:12.5px; opacity:.85; margin-top:8px;
}

.footer{ text-align:center; color:#cbd5e1; padding-top:30px; font-size:14px; opacity:.9; }

/* ============================================================
   SIDEBAR
============================================================ */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#070d1e,#0f1c3d);
    border-right:1px solid rgba(255,255,255,0.06);
}

/* ============================================================
   SCROLLBAR
============================================================ */
::-webkit-scrollbar{ width:8px; }
::-webkit-scrollbar-thumb{ background:#2563eb; border-radius:20px; }

.block-container, section[data-testid="stSidebar"]{ position:relative; z-index:1; }

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE HELPERS
# =========================================================
def init_session_state():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("show_result", False)
    st.session_state.setdefault("do_reset", False)
    st.session_state.setdefault("history", [])


def request_reset():
    """Called from the Reset button. Never touches widget keys directly —
    it only flips a flag and reruns, since widgets in this run are already
    instantiated and their session_state keys can't be changed mid-run."""
    st.session_state["do_reset"] = True
    st.rerun()


def apply_pending_reset():
    """Must run at the very top of the script, before any widget with a
    matching key is created, so it's safe to overwrite those values here."""
    if st.session_state.get("do_reset"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state["show_result"] = False
        st.session_state["do_reset"] = False


init_session_state()
apply_pending_reset()


# =========================================================
# MODEL LOADING (cached)
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_artifacts()


# =========================================================
# CHART BUILDERS
# =========================================================
def salary_gauge(pred: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pred,
        number={"suffix": " LPA", "font": {"size": 40, "color": "white"}},
        delta={
            "reference": (SALARY_MIN + SALARY_MAX) / 2,
            "increasing": {"color": "#22c55e"},
            "decreasing": {"color": "#ef4444"},
        },
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [SALARY_MIN, SALARY_MAX], "tickcolor": "white", "tickfont": {"color": "white"}},
            "bar": {"color": "#3b82f6", "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.2)",
            "steps": [
                {"range": [SALARY_MIN, SALARY_MAX * 0.35], "color": "rgba(239,68,68,0.35)"},
                {"range": [SALARY_MAX * 0.35, SALARY_MAX * 0.65], "color": "rgba(234,179,8,0.35)"},
                {"range": [SALARY_MAX * 0.65, SALARY_MAX], "color": "rgba(34,197,94,0.35)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.9, "value": pred},
        },
    ))
    fig.update_layout(paper_bgcolor=DARK_PAPER, font={"color": DARK_FONT},
                       margin=dict(l=20, r=20, t=30, b=10), height=300)
    return fig


def employee_radar(inputs: dict):
    categories = ["Experience", "Performance", "Skill", "Projects", "Certifications", "Tenure"]
    raw = [
        min(inputs["exp"] / 20, 1) * 100,
        inputs["perf"] / 5 * 100,
        inputs["skill"],
        min(inputs["proj"] / 30, 1) * 100,
        min(inputs["cert"] / 10, 1) * 100,
        min(inputs["tenure"] / 15, 1) * 100,
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=raw + [raw[0]], theta=categories + [categories[0]],
        fill="toself", fillcolor="rgba(59,130,246,0.35)",
        line=dict(color="#60a5fa", width=2), name="Employee Profile",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.03)",
            radialaxis=dict(visible=True, range=[0, 100], color="#93c5fd", gridcolor="rgba(255,255,255,0.15)"),
            angularaxis=dict(color="white", gridcolor="rgba(255,255,255,0.15)"),
        ),
        showlegend=False, paper_bgcolor=DARK_PAPER, font={"color": DARK_FONT},
        margin=dict(l=30, r=30, t=30, b=30), height=340,
    )
    return fig


def build_feature_df(inputs: dict) -> pd.DataFrame:
    """Encodes + scales a raw inputs dict into the exact frame the model expects."""
    df = pd.DataFrame({
        "Age": [inputs["age"]], "Gender": [GENDER_MAP[inputs["gender"]]],
        "Education": [EDUCATION_MAP[inputs["education"]]], "Experience_Years": [inputs["exp"]],
        "Department": [DEPARTMENT_MAP[inputs["dept"]]], "Job_Level": [LEVEL_MAP[inputs["level"]]],
        "Performance_Rating": [inputs["perf"]], "Certifications": [inputs["cert"]],
        "Overtime_Hours": [inputs["over"]], "Remote_Work": [REMOTE_MAP[inputs["remote"]]],
        "City": [CITY_MAP[inputs["city"]]], "Company_Tenure": [inputs["tenure"]],
        "Projects_Completed": [inputs["proj"]], "Skill_Score": [inputs["skill"]],
    })[FEATURE_ORDER]
    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
    return df


def predict_salary(inputs: dict) -> float:
    df = build_feature_df(inputs)
    return float(model.predict(df)[0])


def trajectory_chart(inputs: dict):
    """Salary vs. experience, holding every other input constant — a live 'what if' curve."""
    exp_range = list(range(0, 41, 2))
    preds = []
    for e in exp_range:
        temp = dict(inputs)
        temp["exp"] = float(e)
        preds.append(predict_salary(temp))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=exp_range, y=preds, mode="lines", fill="tozeroy",
        line=dict(color="#60a5fa", width=3, shape="spline"),
        fillcolor="rgba(59,130,246,0.18)", name="Projected Salary",
    ))
    fig.add_trace(go.Scatter(
        x=[inputs["exp"]], y=[predict_salary(inputs)], mode="markers",
        marker=dict(color="#22c55e", size=13, line=dict(color="white", width=2)),
        name="You are here",
    ))
    fig.update_layout(
        paper_bgcolor=DARK_PAPER, plot_bgcolor=DARK_PAPER, font={"color": DARK_FONT},
        xaxis=dict(title="Experience (Years)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Salary (LPA)", gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(l=10, r=10, t=20, b=10), height=300, showlegend=False,
    )
    return fig


def model_convergence_chart(df: pd.DataFrame):
    """
    Real model behavior (not illustrative): shows the GBR's actual prediction after
    each boosting stage, via staged_predict(), so you can see the ensemble converge
    to its final answer as more trees are added.
    """
    stage_preds = [float(p[0]) for p in model.staged_predict(df)]
    final = stage_preds[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(stage_preds) + 1)), y=stage_preds,
        mode="lines", line=dict(color="#60a5fa", width=2.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.15)", name="Running estimate",
    ))
    fig.add_hline(y=final, line_width=2, line_dash="dash", line_color="#22c55e")
    fig.update_layout(
        paper_bgcolor=DARK_PAPER, plot_bgcolor=DARK_PAPER, font={"color": DARK_FONT},
        xaxis=dict(title="Boosting stage (tree #)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Running Salary Estimate (LPA)", gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(l=10, r=10, t=20, b=10), height=280, showlegend=False,
    )
    return fig


def department_comparison_chart(inputs: dict):
    """
    Real 'what if' comparison: holds every other input fixed and re-runs the model
    for each department, so you can see the actual effect of that one factor.
    """
    depts = list(DEPARTMENT_MAP)
    preds = []
    for d in depts:
        temp = dict(inputs)
        temp["dept"] = d
        preds.append(predict_salary(temp))

    colors = ["#22c55e" if d == inputs["dept"] else "#3b82f6" for d in depts]

    fig = go.Figure(go.Bar(x=depts, y=preds, marker=dict(color=colors)))
    fig.update_layout(
        paper_bgcolor=DARK_PAPER, plot_bgcolor=DARK_PAPER, font={"color": DARK_FONT},
        xaxis=dict(title="Department", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Predicted Salary (LPA)", gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(l=10, r=10, t=20, b=10), height=280, showlegend=False,
    )
    return fig


def feature_importance_chart(model, feature_order):
    """Live bar chart of the loaded model's real feature_importances_."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None
    pairs = sorted(zip(feature_order, importances), key=lambda x: x[1])
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=vals, colorscale=[[0, "#1e3a8a"], [0.5, "#3b82f6"], [1, "#60a5fa"]]),
    ))
    fig.update_layout(
        paper_bgcolor=DARK_PAPER, plot_bgcolor=DARK_PAPER, font={"color": DARK_FONT},
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", title="Relative Importance"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=10, r=10, t=20, b=10), height=420,
    )
    return fig


# =========================================================
# UI COMPONENTS
# =========================================================
def hero():
    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <div class="money">💰</div>
            <h1>Employee Salary Predictor</h1>
            <div class="hero-subtitle">
                <span class="hero-pill">✨ AI-Powered Salary Prediction</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def sidebar_nav():
    with st.sidebar:
        st.title("🧭 Navigation")
        st.markdown("---")
        page = st.radio(
            "Go to",
            ["🏠 Predictor", "🕘 History", "ℹ️ About", "📖 Model Info", "📧 Contact"],
            label_visibility="collapsed", key="nav_page",
        )
        st.markdown("---")
        st.info(f"""
### SalarySense AI

Predict employee salaries using Machine Learning.

Version **{VERSION}**
""")
        st.markdown("---")
        st.success("Gradient Boosting Regressor")
        st.metric("R² Score", "0.775")
        st.metric("Developer", DEVELOPER)
        st.markdown("---")
        st.caption(f"{APP_TITLE} · v{VERSION}")
    return page


def employee_summary_panel(values: dict):
    rows = [
        ("Age", values["age"]), ("Gender", values["gender"]), ("Education", values["education"]),
        ("City", values["city"]), ("Department", values["dept"]), ("Job Level", values["level"]),
        ("Remote Work", values["remote"]), ("Experience (yrs)", values["exp"]),
        ("Company Tenure (yrs)", values["tenure"]), ("Certifications", values["cert"]),
        ("Overtime Hours", values["over"]), ("Projects Completed", values["proj"]),
        ("Performance Rating", f'{values["perf"]} / 5'), ("Skill Score", f'{values["skill"]} / 100'),
    ]
    items_html = "".join(
        f'<div class="summary-item"><span>{label}</span><span>{val}</span></div>'
        for label, val in rows
    )
    st.markdown(f"""
    <div class="glass">
        <div class="section-title">👤 Employee Summary</div>
        {items_html}
    </div>
    """, unsafe_allow_html=True)


def footer():
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        Built with ❤️ using <b>Python • Streamlit • Scikit-learn • Plotly</b>
        <br><br>
        © 2026 Naved Alam
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# SECONDARY PAGES
# =========================================================
def render_about():
    st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
        <p><b>SalarySense AI</b> is a machine-learning powered tool that estimates an employee's
        expected annual salary (in LPA) based on profile, experience, and performance factors.</p>
        <p>The goal is to give HR teams, recruiters, and candidates a fast, data-driven reference
        point for compensation benchmarking — not a substitute for a full compensation review.</p>
        <span class="badge">Python</span>
        <span class="badge">Streamlit</span>
        <span class="badge">Scikit-learn</span>
        <span class="badge">Plotly</span>
        <span class="badge">Gradient Boosting</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass">
            <div class="section-title">🎯 What it does</div>
            <ul>
                <li>Takes 14 employee attributes as input</li>
                <li>Scales numeric features the same way as during training</li>
                <li>Runs them through a trained Gradient Boosting Regressor</li>
                <li>Returns an estimated annual salary with a visual gauge</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="glass">
            <div class="section-title">🚀 Version History</div>
            <ul>
                <li><b>v1.0</b> — Initial prediction form</li>
                <li><b>v2.0</b> — Styled UI with hero + result card</li>
                <li><b>v{VERSION}</b> — Glassmorphism UI, gauge & radar charts,
                    multi-page navigation, reset button</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def render_model_info(model):
    st.markdown('<div class="section-title">📖 Model Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Algorithm", "Gradient Boosting")
    with c2:
        st.metric("R² Score", "0.775")
    with c3:
        st.metric("Estimators", getattr(model, "n_estimators", "—"))

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="section-title">📊 Feature Importance</div>', unsafe_allow_html=True)
        fig = feature_importance_chart(model, FEATURE_ORDER)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("This model does not expose feature_importances_.")

    st.markdown(f"""
    <div class="glass">
        <div class="section-title">🧾 Input Features</div>
        <p>The model expects the following {len(FEATURE_ORDER)} features, in this order:</p>
        {"".join(f'<span class="badge">{f}</span>' for f in FEATURE_ORDER)}
    </div>
    """, unsafe_allow_html=True)


def render_contact():
    st.markdown('<div class="section-title">📧 Contact</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown(f"""
        <div class="glass">
            <div class="section-title">👋 Get in touch</div>
            <p>Built and maintained by <b>{DEVELOPER}</b>.</p>
            <span class="badge">📧 Email</span>
            <span class="badge">💼 LinkedIn</span>
            <span class="badge">🐙 GitHub</span>
            <p style="margin-top:14px; opacity:.85; font-size:13px;">
            Replace these placeholders with your real profile links before deploying.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown('<div class="section-title">✉️ Send a message</div>', unsafe_allow_html=True)
            with st.form("contact_form", clear_on_submit=True):
                name = st.text_input("Your Name")
                email = st.text_input("Your Email")
                message = st.text_area("Message", height=120)
                sent = st.form_submit_button("📨 Send")
                if sent:
                    if name and email and message:
                        st.success("Thanks! Your message has been noted. (Demo form — connect a backend to make this live.)")
                    else:
                        st.warning("Please fill in all fields before sending.")


def render_history():
    st.markdown('<div class="section-title">🕘 Prediction History</div>', unsafe_allow_html=True)
    history = st.session_state.get("history", [])

    if not history:
        st.markdown("""
        <div class="glass">
            <p>No predictions yet this session. Head to the <b>Predictor</b> tab, fill in a
            profile, and click <b>Predict Salary</b> — each result you generate will show up here.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    hist_df = pd.DataFrame(history)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predictions Made", len(hist_df))
    with c2:
        st.metric("Average", f"₹{hist_df['Predicted Salary (LPA)'].mean():,.2f} LPA")
    with c3:
        st.metric("Highest", f"₹{hist_df['Predicted Salary (LPA)'].max():,.2f} LPA")

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="section-title">📈 Trend This Session</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(
            x=list(range(1, len(hist_df) + 1)), y=hist_df["Predicted Salary (LPA)"],
            mode="lines+markers", line=dict(color="#60a5fa", width=3),
            marker=dict(color="#22c55e", size=8, line=dict(color="white", width=1)),
        ))
        fig.update_layout(
            paper_bgcolor=DARK_PAPER, plot_bgcolor=DARK_PAPER, font={"color": DARK_FONT},
            xaxis=dict(title="Prediction #", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="Salary (LPA)", gridcolor="rgba(255,255,255,0.08)"),
            margin=dict(l=10, r=10, t=20, b=10), height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">🧾 All Predictions</div>', unsafe_allow_html=True)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

    d1, d2 = st.columns([1, 1])
    with d1:
        st.download_button(
            "⬇️ Download as CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
            st.session_state["history"] = []
            st.rerun()


# =========================================================
# MAIN APP FLOW
# =========================================================
page = sidebar_nav()

hero()

if page == "🏠 Predictor":

    with st.container(border=True):
        st.markdown('<div class="section-title">👤 Employee Profile</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("🎂 Age", 18, 70, key="age")
            gender = st.selectbox("👤 Gender", list(GENDER_MAP), key="gender")
            education = st.selectbox("🎓 Education", list(EDUCATION_MAP), key="education")
            city = st.selectbox("🌍 City", list(CITY_MAP), key="city")
        with c2:
            dept = st.selectbox("🏢 Department", list(DEPARTMENT_MAP), key="dept")
            level = st.selectbox("📈 Job Level", list(LEVEL_MAP), key="level")
            remote = st.selectbox("🏠 Remote Work", list(REMOTE_MAP), key="remote")

        st.markdown('<div class="section-title">💼 Work & Performance</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            exp = st.number_input("💼 Experience (Years)", 0.0, 50.0, key="exp")
            tenure = st.number_input("📅 Company Tenure", 0.0, 50.0, key="tenure")
            cert = st.number_input("📜 Certifications", 0, 20, key="cert")
            over = st.number_input("⏰ Overtime Hours", 0, 200, key="over")
        with c4:
            proj = st.number_input("📂 Projects Completed", 0, 100, key="proj")
            perf = st.slider("⭐ Performance Rating", 1, 5, key="perf")
            skill = st.slider("🧠 Skill Score", 0, 100, key="skill")

    inputs = {
        "age": age, "gender": gender, "education": education, "city": city,
        "dept": dept, "level": level, "remote": remote, "exp": exp,
        "tenure": tenure, "cert": cert, "over": over, "proj": proj,
        "perf": perf, "skill": skill,
    }

    # --- live, reactive estimate (recomputes on every widget change, no click needed) ---
    live_pred = predict_salary(inputs)
    st.markdown(f"""
    <div class="live-chip">
        <div class="live-label"><span class="live-dot"></span> Live Estimate (updates as you type)</div>
        <div class="live-amount">₹ {live_pred:,.2f} LPA</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">📈 Career Growth Trajectory</div>', unsafe_allow_html=True)
        st.plotly_chart(trajectory_chart(inputs), use_container_width=True)
        st.markdown(
            '<div class="influence-note">Shows how the estimate would change with more or less '
            'experience, holding everything else fixed at your current selections.</div>',
            unsafe_allow_html=True,
        )

    b1, b2 = st.columns([3, 1])
    with b1:
        predict_clicked = st.button("🚀 Predict Salary", use_container_width=True)
    with b2:
        if st.button("🔄 Reset", use_container_width=True, type="secondary"):
            request_reset()

    if predict_clicked:
        pred = predict_salary(inputs)
        st.session_state["show_result"] = True
        st.session_state["last_pred"] = pred
        st.session_state["last_inputs"] = dict(inputs)

        st.session_state["history"].append({
            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
            "Predicted Salary (LPA)": round(pred, 2),
            "Age": age, "Department": dept, "Job Level": level,
            "Experience (yrs)": exp, "Skill Score": skill,
        })

        st.balloons()
        st.toast(f"Prediction complete: ₹{pred:,.2f} LPA", icon="✅")

    if st.session_state.get("show_result"):

        placeholder = st.empty()
        placeholder.markdown("""
        <div class="loader-wrap">
            <div class="loader-dot"></div><div class="loader-dot"></div><div class="loader-dot"></div>
            &nbsp; Crunching the numbers...
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.7)
        placeholder.empty()

        # Use the frozen values from the moment "Predict" was clicked, so navigating
        # away/back or nudging a slider afterward doesn't silently change the result.
        pred = st.session_state["last_pred"]
        result_inputs = st.session_state["last_inputs"]
        r_exp, r_level, r_dept = result_inputs["exp"], result_inputs["level"], result_inputs["dept"]

        st.markdown(f"""
        <div class="result-card">
            <div class="label">💰 Estimated Annual Salary</div>
            <div class="amount">₹ {pred:,.2f} LPA</div>
            <div class="sub">Based on {r_exp:.1f} yrs experience · {r_level} level · {r_dept} dept</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="section-title">🏢 Department Comparison</div>', unsafe_allow_html=True)
            st.plotly_chart(department_comparison_chart(result_inputs), use_container_width=True)
            st.markdown(
                '<div class="influence-note">Same profile, re-scored in every department — '
                'your current department is highlighted in green.</div>',
                unsafe_allow_html=True,
            )

        employee_summary_panel(result_inputs)

    st.caption(f"Developed by {DEVELOPER}")

elif page == "🕘 History":
    render_history()

elif page == "ℹ️ About":
    render_about()

elif page == "📖 Model Info":
    render_model_info(model)

elif page == "📧 Contact":
    render_contact()

footer()