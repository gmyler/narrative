
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Narrative Intelligence Engine",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.035em;
    }

    .hero {
        padding: 1.5rem 1.6rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #062f2f 0%, #0f766e 52%, #facc15 150%);
        color: white;
        margin-bottom: 1.1rem;
        box-shadow: 0 22px 55px rgba(15, 118, 110, 0.26);
    }

    .hero-title {
        font-size: 2.55rem;
        font-weight: 900;
        line-height: 1.02;
        margin-bottom: 0.45rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #ecfffb;
        max-width: 1050px;
    }

    .card {
        background: white;
        border: 1px solid #d8ece9;
        border-radius: 24px;
        padding: 1.05rem 1.1rem;
        box-shadow: 0 12px 34px rgba(15, 118, 110, 0.08);
        margin-bottom: 1rem;
    }

    .metric-card {
        min-height: 118px;
        background: white;
        border: 1px solid #d8ece9;
        border-radius: 24px;
        padding: 1.05rem 1.1rem;
        box-shadow: 0 12px 34px rgba(15, 118, 110, 0.08);
    }

    .metric-label {
        font-size: 0.75rem;
        color: #637b78;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.75rem;
        color: #063c3a;
        font-weight: 900;
        line-height: 1.05;
    }

    .metric-note {
        margin-top: 0.35rem;
        color: #6b827f;
        font-size: 0.84rem;
    }

    .tag {
        display: inline-block;
        border-radius: 999px;
        padding: 0.32rem 0.62rem;
        background: #e7f7f4;
        color: #08756f;
        font-size: 0.76rem;
        font-weight: 800;
        margin-right: 0.28rem;
        margin-bottom: 0.3rem;
    }

    .tag-dark {
        display: inline-block;
        border-radius: 999px;
        padding: 0.32rem 0.62rem;
        background: #083d3b;
        color: #ffffff;
        font-size: 0.76rem;
        font-weight: 800;
        margin-right: 0.28rem;
        margin-bottom: 0.3rem;
    }

    .warning-box {
        background: #fff8e8;
        border: 1px solid #ffd98a;
        color: #604200;
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin: 0.65rem 0 1rem 0;
    }

    .evidence-card {
        background: white;
        border: 1px solid #d8ece9;
        border-left: 6px solid #0f766e;
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 8px 24px rgba(15, 118, 110, 0.07);
    }

    .small-muted {
        color: #6d827f;
        font-size: 0.88rem;
    }

    .formula {
        background: #f4fbfa;
        border: 1px solid #d7efeb;
        border-radius: 18px;
        padding: 1rem;
        color: #08413e;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        white-space: pre-wrap;
        margin-bottom: 1rem;
    }

    .section-note {
        font-size: 0.95rem;
        color: #49615f;
        margin-bottom: 0.7rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"


def find_file(filename: str) -> Path | None:
    candidates = [
        DATA_DIR / filename,
        APP_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame:
    path = find_file(filename)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(filename: str) -> dict:
    path = find_file(filename)
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


articles = load_csv("article_scores_final.csv")
clusters = load_csv("narrative_cluster_summary_final.csv")
actors = load_csv("actor_scores_showcase.csv")
agenda = load_csv("agenda_shapers_showcase.csv")
outlets = load_csv("outlet_scores_final.csv")
evidence = load_csv("evidence_snippets_showcase.csv")
positions = load_csv("debate_position_table_showcase.csv")
overview = load_json("overview_showcase.json")


def fmt_num(x):
    try:
        x = float(x)
    except Exception:
        return "0"

    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f}K"
    if x == int(x):
        return f"{int(x):,}"
    return f"{x:,.1f}"


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stance_bucket(score):
    try:
        score = float(score)
    except Exception:
        return "Mixed / unclear"

    if score <= -0.45:
        return "Strong preserve"
    if score <= -0.15:
        return "Lean preserve"
    if score < 0.15:
        return "Mixed / unclear"
    if score < 0.45:
        return "Reform / redefine"
    return "Move away / align"


def source_link(url):
    if isinstance(url, str) and url.strip() and url.lower() != "nan":
        return f"<a href='{url}' target='_blank'>Open source</a>"
    return ""


def display_table(df, cols, n=None):
    if df.empty:
        st.info("No data available.")
        return
    use_cols = [c for c in cols if c in df.columns]
    out = df[use_cols].copy()
    if n:
        out = out.head(n)
    st.dataframe(out, use_container_width=True, hide_index=True)


def get_cluster_col(df):
    if "cluster_display_name" in df.columns:
        return "cluster_display_name"
    if "cluster_name" in df.columns:
        return "cluster_name"
    return None


# Sidebar
st.sidebar.markdown("## 🧭 Narrative Engine")
st.sidebar.markdown("CSV-powered demo dashboard")

page = st.sidebar.radio(
    "View",
    [
        "Executive Overview",
        "Narrative Map",
        "Agenda Shapers",
        "Debate Positioning",
        "Outlet Influence",
        "Evidence Explorer",
        "Impact Lab",
        "Methodology",
    ],
)

st.sidebar.markdown("---")

cluster_filter = "All"
cluster_col_articles = get_cluster_col(articles)
if not articles.empty and cluster_col_articles:
    cluster_options = ["All"] + sorted(articles[cluster_col_articles].dropna().astype(str).unique().tolist())
    cluster_filter = st.sidebar.selectbox("Narrative cluster", cluster_options)

stance_filter = "All"
if not articles.empty and "stance_bucket" in articles.columns:
    stance_options = ["All"] + sorted(articles["stance_bucket"].dropna().astype(str).unique().tolist())
    stance_filter = st.sidebar.selectbox("Stance", stance_options)

outlet_filter = "All"
if not articles.empty and "Outlet" in articles.columns:
    outlet_options = ["All"] + sorted(articles["Outlet"].dropna().astype(str).unique().tolist())
    outlet_filter = st.sidebar.selectbox("Outlet", outlet_options)

filtered_articles = articles.copy()

if cluster_filter != "All" and cluster_col_articles:
    filtered_articles = filtered_articles[filtered_articles[cluster_col_articles] == cluster_filter]

if stance_filter != "All" and "stance_bucket" in filtered_articles.columns:
    filtered_articles = filtered_articles[filtered_articles["stance_bucket"] == stance_filter]

if outlet_filter != "All" and "Outlet" in filtered_articles.columns:
    filtered_articles = filtered_articles[filtered_articles["Outlet"] == outlet_filter]


# Header
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Narrative Intelligence Engine</div>
        <div class="hero-subtitle">
            A demo intelligence layer for clustering media coverage, weighting outlet influence,
            mapping debate stance, ranking agenda shapers, and drilling into evidence.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if overview.get("date_warning"):
    st.markdown(
        f"""
        <div class="warning-box">
            <b>Important caveat:</b> {overview.get("date_warning")}
        </div>
        """,
        unsafe_allow_html=True,
    )

if articles.empty and clusters.empty and agenda.empty:
    st.error("No data files were found. Put the CSV/JSON files either in the repo root or in a data/ folder.")
    st.stop()


# Executive Overview
if page == "Executive Overview":
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card("Mentions analysed", fmt_num(len(articles)), "filtered coverage items")
    with c2:
        metric_card(
            "Narrative clusters",
            fmt_num(clusters["cluster_display_name"].nunique() if "cluster_display_name" in clusters.columns else len(clusters)),
            "semantic subtopics",
        )
    with c3:
        metric_card("Agenda shapers", fmt_num(len(agenda)), "actors + outlets")
    with c4:
        metric_card("Media outlets", fmt_num(outlets["Outlet"].nunique() if "Outlet" in outlets.columns else 0), "source universe")
    with c5:
        top_narrative = overview.get("top_narrative_cluster", "")
        if not top_narrative and not clusters.empty:
            top_narrative = clusters.sort_values("weighted_mentions", ascending=False).iloc[0].get("cluster_display_name", "")
        metric_card("Top narrative", str(top_narrative)[:24] + ("…" if len(str(top_narrative)) > 24 else ""), "by weighted volume")

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Narrative cluster landscape")
        st.markdown(
            "<div class='section-note'>Bubble size = influence-weighted coverage. Colour = stance direction.</div>",
            unsafe_allow_html=True,
        )

        if not clusters.empty:
            plot_df = clusters.copy()
            if "avg_stance_score" not in plot_df.columns:
                plot_df["avg_stance_score"] = 0
            if "weighted_mentions" not in plot_df.columns:
                plot_df["weighted_mentions"] = 1
            if "mention_count" not in plot_df.columns:
                plot_df["mention_count"] = 1
            plot_df["stance_direction"] = plot_df["avg_stance_score"].apply(stance_bucket)
            plot_df["bubble_size"] = pd.to_numeric(plot_df["weighted_mentions"], errors="coerce").fillna(1)

            fig = px.scatter(
                plot_df,
                x="mention_count",
                y="weighted_mentions",
                size="bubble_size",
                color="stance_direction",
                hover_name="cluster_display_name",
                hover_data=[c for c in ["mention_count", "weighted_mentions", "estimated_reach", "avg_stance_score", "dominant_stance", "dominant_frame"] if c in plot_df.columns],
                template="plotly_white",
                size_max=60,
                labels={
                    "mention_count": "Mention volume",
                    "weighted_mentions": "Influence-weighted volume",
                    "stance_direction": "Stance direction",
                },
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cluster file found.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Top agenda shapers")
        display_table(
            agenda.sort_values("impact_score", ascending=False) if "impact_score" in agenda.columns else agenda,
            ["entity", "entity_group", "actor_type", "narrative_role", "stance_label", "impact_score"],
            12,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Stance distribution")
        if not filtered_articles.empty and "stance_bucket" in filtered_articles.columns:
            stance_df = filtered_articles["stance_bucket"].value_counts().reset_index()
            stance_df.columns = ["stance", "mentions"]
            fig = px.bar(
                stance_df,
                x="mentions",
                y="stance",
                orientation="h",
                template="plotly_white",
                labels={"mentions": "Mentions", "stance": ""},
            )
            fig.update_layout(height=320, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No stance data found.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("What this demo proves")
    st.write(
        """
        The tool moves beyond basic sentiment. It separates **sentiment**, **stance**, **frame**,
        **influence**, **actor role**, and **evidence**. Coverage can look neutral in tone while still
        shifting a debate through repeated framing, actor prominence, and outlet influence.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Narrative Map":
    st.header("Narrative Map")
    st.write("Explore the main subtopics and coverage clusters inside the debate.")

    if clusters.empty:
        st.warning("No cluster data found.")
    else:
        sort_col = "weighted_mentions" if "weighted_mentions" in clusters.columns else "mention_count"
        clusters_sorted = clusters.sort_values(sort_col, ascending=False)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = px.bar(
            clusters_sorted,
            x=sort_col,
            y="cluster_display_name",
            color="avg_stance_score" if "avg_stance_score" in clusters_sorted.columns else None,
            orientation="h",
            hover_data=[c for c in ["mention_count", "estimated_reach", "dominant_stance", "dominant_frame"] if c in clusters_sorted.columns],
            template="plotly_white",
            labels={sort_col: "Weighted coverage", "cluster_display_name": "", "avg_stance_score": "Avg stance"},
        )
        fig.update_layout(height=620, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        selected_cluster = st.selectbox("Open a narrative cluster", clusters_sorted["cluster_display_name"].tolist())

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader(selected_cluster)
        sub = articles[articles[cluster_col_articles] == selected_cluster] if cluster_col_articles else pd.DataFrame()

        a, b, c, d = st.columns(4)
        with a:
            metric_card("Mentions", fmt_num(len(sub)))
        with b:
            metric_card("Weighted volume", fmt_num(sub["mention_weight"].sum() if "mention_weight" in sub.columns else 0))
        with c:
            metric_card("Estimated reach", fmt_num(sub["Estimated Reach"].sum() if "Estimated Reach" in sub.columns else 0))
        with d:
            avg = sub["stance_score"].mean() if "stance_score" in sub.columns and len(sub) else 0
            metric_card("Avg stance", f"{avg:.2f}", stance_bucket(avg))

        st.subheader("Top articles in this narrative")
        if not sub.empty:
            rank = "mention_weight" if "mention_weight" in sub.columns else None
            if rank:
                sub = sub.sort_values(rank, ascending=False)
            display_table(
                sub,
                ["Headline", "Outlet", "Sentiment", "Estimated Reach", "stance_bucket", "dominant_frame", "mention_weight_100", "Link"],
                30,
            )
        else:
            st.info("No articles found for this cluster.")
        st.markdown("</div>", unsafe_allow_html=True)


elif page == "Agenda Shapers":
    st.header("Agenda Shapers")
    st.write("Who is shaping the debate, what role are they playing, and where are they positioned?")

    if agenda.empty:
        st.warning("No agenda shaper data found.")
    else:
        group_filter = st.radio("Entity type", ["All", "Actor / organisation", "Media outlet"], horizontal=True)
        agenda_view = agenda.copy()
        if group_filter != "All" and "entity_group" in agenda_view.columns:
            agenda_view = agenda_view[agenda_view["entity_group"] == group_filter]

        left, right = st.columns([1.45, 1])

        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            plot_df = agenda_view.copy()
            if "avg_stance" not in plot_df.columns:
                plot_df["avg_stance"] = 0
            if "impact_score" not in plot_df.columns:
                plot_df["impact_score"] = 0
            if "weighted_mentions" not in plot_df.columns:
                plot_df["weighted_mentions"] = 1

            fig = px.scatter(
                plot_df.head(80),
                x="avg_stance",
                y="impact_score",
                size="weighted_mentions",
                color="narrative_role" if "narrative_role" in plot_df.columns else None,
                hover_name="entity",
                hover_data=[c for c in ["entity_group", "actor_type", "mentions", "unique_stories", "top_cluster", "stance_label"] if c in plot_df.columns],
                template="plotly_white",
                labels={
                    "avg_stance": "Debate position: preserve ← → move away",
                    "impact_score": "Impact score",
                    "weighted_mentions": "Weighted mentions",
                },
                size_max=44,
            )
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=600, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            selected = st.selectbox("Open contributor profile", agenda_view["entity"].astype(str).tolist())
            row = agenda_view[agenda_view["entity"].astype(str) == selected].iloc[0]

            st.subheader(selected)
            tag_html = ""
            for tag in [row.get("entity_group"), row.get("actor_type"), row.get("narrative_role"), row.get("stance_label")]:
                if pd.notna(tag):
                    tag_html += f"<span class='tag'>{tag}</span>"
            st.markdown(tag_html, unsafe_allow_html=True)

            a, b = st.columns(2)
            with a:
                metric_card("Impact score", fmt_num(row.get("impact_score", 0)))
            with b:
                metric_card("Share of voice", f"{float(row.get('share_of_voice', 0)) * 100:.2f}%")

            c, d = st.columns(2)
            with c:
                metric_card("Mentions", fmt_num(row.get("mentions", 0)))
            with d:
                metric_card("Reach", fmt_num(row.get("estimated_reach", 0)))

            st.markdown("**Interpretation**")
            st.write(
                f"""
                **Role:** {row.get("narrative_role", "")}  
                **Stance:** {row.get("stance_label", "")}  
                **Top narrative:** {row.get("top_cluster", "")}  
                **Top frame:** {row.get("top_frame", "")}
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Full agenda shaper table")
        display_table(
            agenda_view.sort_values("impact_score", ascending=False) if "impact_score" in agenda_view.columns else agenda_view,
            ["entity", "entity_group", "actor_type", "narrative_role", "impact_score", "mentions", "unique_stories", "weighted_mentions", "estimated_reach", "stance_label", "top_cluster", "top_frame"],
        )
        st.markdown("</div>", unsafe_allow_html=True)


elif page == "Debate Positioning":
    st.header("Debate Positioning")
    st.write("Maps who is associated with preserving the current position, reforming it, or moving away.")

    if positions.empty:
        st.warning("No debate position table found.")
    else:
        plot_df = positions.copy()
        if "avg_stance" not in plot_df.columns:
            plot_df["avg_stance"] = 0
        if "impact_score" not in plot_df.columns:
            plot_df["impact_score"] = 0

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = px.scatter(
            plot_df,
            x="avg_stance",
            y="impact_score",
            size="weighted_mentions" if "weighted_mentions" in plot_df.columns else None,
            color="debate_position" if "debate_position" in plot_df.columns else "stance_label",
            hover_name="actor",
            hover_data=[c for c in ["actor_type", "narrative_role", "top_cluster", "top_frame"] if c in plot_df.columns],
            template="plotly_white",
            labels={
                "avg_stance": "Preserve / stay ←     → reform / move away",
                "impact_score": "Impact score",
            },
            size_max=44,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Position table")
        display_table(
            plot_df.sort_values("avg_stance"),
            ["actor", "actor_type", "debate_position", "avg_stance", "impact_score", "narrative_role", "top_cluster", "top_frame"],
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.info("Interpretation note: this is the stance of coverage associated with the actor, not a definitive claim about personal belief.")


elif page == "Outlet Influence":
    st.header("Outlet Influence")
    st.write("Which outlets carried the most influence-weighted coverage?")

    if outlets.empty:
        st.warning("No outlet data found.")
    else:
        sort = "impact_score" if "impact_score" in outlets.columns else "weighted_mentions"
        view = outlets.sort_values(sort, ascending=False)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = px.bar(
            view.head(30),
            x=sort,
            y="Outlet",
            color="avg_stance" if "avg_stance" in view.columns else None,
            orientation="h",
            hover_data=[c for c in ["mentions", "unique_stories", "weighted_mentions", "estimated_reach", "top_cluster", "top_frame"] if c in view.columns],
            template="plotly_white",
            labels={sort: "Outlet impact", "Outlet": "", "avg_stance": "Avg stance"},
        )
        fig.update_layout(height=720, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Outlet table")
        display_table(
            view,
            ["Outlet", "impact_score", "mentions", "unique_stories", "weighted_mentions", "estimated_reach", "avg_stance", "avg_sentiment", "top_cluster", "top_frame", "top_tier"],
        )
        st.markdown("</div>", unsafe_allow_html=True)


elif page == "Evidence Explorer":
    st.header("Evidence Explorer")
    st.write("Click into the source evidence behind narratives, actors, outlets and scores.")

    if evidence.empty:
        st.warning("No evidence snippets found.")
    else:
        entities = ["All"] + sorted(evidence["entity"].dropna().astype(str).unique().tolist())
        selected_entity = st.selectbox("Entity / outlet", entities)

        ev = evidence.copy()

        if selected_entity != "All":
            ev = ev[ev["entity"].astype(str) == selected_entity]

        if cluster_filter != "All" and "cluster_name" in ev.columns:
            ev = ev[ev["cluster_name"] == cluster_filter]

        st.write(f"Showing **{len(ev)}** evidence snippets.")

        for _, row in ev.head(100).iterrows():
            st.markdown(
                f"""
                <div class="evidence-card">
                    <b>{row.get("headline", "")}</b><br>
                    <span class="small-muted">
                        {row.get("outlet", "")} · {row.get("cluster_name", "")} · {row.get("stance_bucket", "")}
                    </span>
                    <p>{row.get("evidence_excerpt", "")}</p>
                    <span class="tag-dark">{row.get("entity", "")}</span>
                    <span class="tag">{row.get("dominant_frame", "")}</span>
                    <br>{source_link(row.get("link", ""))}
                </div>
                """,
                unsafe_allow_html=True,
            )


elif page == "Impact Lab":
    st.header("Impact Lab")
    st.write("Separates current agenda impact from true intervention/reframing impact.")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("1. Agenda / Media Impact Score")
        st.write("This is available in the current dataset. It answers: **who is most influential in the debate right now?**")
        st.markdown(
            """
            <div class="formula">Agenda Impact Score =
0.30 × Weighted mentions
+ 0.23 × Estimated reach
+ 0.20 × Unique stories
+ 0.14 × Cluster spread
+ 0.13 × Stance strength</div>
            """,
            unsafe_allow_html=True,
        )
        display_table(
            agenda.sort_values("impact_score", ascending=False) if "impact_score" in agenda.columns else agenda,
            ["entity", "entity_group", "narrative_role", "stance_label", "impact_score", "weighted_mentions", "estimated_reach", "unique_stories"],
            15,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("2. Intervention Impact Score")
        st.write("This is the premium module. It answers: **did an intervention reframe the conversation afterwards?**")
        st.markdown(
            """
            <div class="formula">Intervention Impact =
0.20 × Reach Shift
+ 0.25 × Narrative Adoption
+ 0.20 × Frame Shift
+ 0.15 × Actor Response
+ 0.10 × Stance Movement
+ 0.10 × Persistence</div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            "To calculate this rigorously, the dataset needs clean publication dates and a list of interventions/events. "
            "This demo currently shows the design and the agenda-impact layer."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("What extra fields unlock full intervention impact?")
    st.write(
        """
        Required fields:
        - `publication_date`
        - `intervention_date`
        - `intervention_actor`
        - `intervention_message`
        - `intervention_type`
        - enough articles before and after the intervention window

        Once those exist, the app can compare pre/post coverage, test whether later articles become semantically closer
        to the intervention message, measure whether the stance direction changed, and identify which actors responded.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Methodology":
    st.header("Methodology")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("What the model layer does")
    st.write(
        """
        This demo uses precomputed model outputs. The app itself reads CSVs, which makes it fast and stable for presentation.

        The model layer does six things:

        1. **Narrative clustering**: groups coverage into subtopic clusters.
        2. **Influence weighting**: weights coverage by reach, outlet tier, article classification and centrality.
        3. **Stance mapping**: maps coverage onto a topic-specific debate axis.
        4. **Sentiment/frame separation**: distinguishes tone from the actual narrative frame.
        5. **Agenda shaper ranking**: identifies actors, organisations and media outlets shaping the debate.
        6. **Evidence drilldown**: preserves excerpts and links so each score can be inspected.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Why this is not just sentiment analysis")
    st.write(
        """
        Standard sentiment can say coverage is mostly neutral. But neutral tone can still be strategically important.
        A neutral article can repeatedly connect an issue to security, cost, sovereignty, moral duty, public consent,
        or institutional failure. This app therefore shows **stance**, **frame**, **role** and **influence**, not just positive/negative tone.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Current limitation")
    st.write(
        """
        The current sample does not include a clean publication date field. That means the dashboard can show agenda/media impact,
        but true pre/post intervention impact should be treated as a designed module until clean dates and intervention records are supplied.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
