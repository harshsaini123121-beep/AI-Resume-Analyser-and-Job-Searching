"""
Reusable Plotly Chart Components:
- ATS Category Horizontal Bar Chart
- Domain Skill Radar / Spider Chart
- Grouped Match Signal Contribution Bar Chart
"""
from typing import Optional, Dict, List, Any
import html
import plotly.graph_objects as go


def render_ats_category_chart(category_breakdown: Optional[Dict[str, Any]]) -> Optional[go.Figure]:
    """Renders accessible horizontal bar chart for the 8 ATS categories."""
    if not category_breakdown:
        return None

    categories = []
    percentages = []
    scores = []
    max_scores = []
    colors = []
    hover_texts = []

    for cat_name, info in category_breakdown.items():
        score = info.get("score", 0)
        max_s = info.get("max_score", 10)
        pct = int((score / max_s) * 100) if max_s > 0 else 0
        details = info.get("details", "")

        categories.append(html.escape(str(cat_name)))
        scores.append(score)
        max_scores.append(max_s)
        percentages.append(pct)

        if pct >= 80:
            colors.append("#10b981")  # Emerald
        elif pct >= 60:
            colors.append("#2563eb")  # Blue
        elif pct >= 40:
            colors.append("#f59e0b")  # Amber
        else:
            colors.append("#ef4444")  # Red

        hover_texts.append(f"<b>{cat_name}</b><br>Score: {score}/{max_s} pts ({pct}%)<br>{details}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories,
        x=percentages,
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255, 255, 255, 0.7)', width=1)),
        text=[f"{p}% ({s}/{m})" for p, s, m in zip(percentages, scores, max_scores)],
        textposition='inside',
        textfont=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', size=11, color='#ffffff'),
        hovertext=hover_texts,
        hoverinfo='text'
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor='#f1f5f9',
            tickfont=dict(size=11, color='#64748b'),
            ticksuffix='%'
        ),
        yaxis=dict(
            autorange='reversed',
            tickfont=dict(size=12, color='#0f172a')
        )
    )
    return fig


def render_skill_radar_chart(categories: Optional[Dict[str, Any]], candidate_skills: Optional[List[str]]) -> Optional[go.Figure]:
    """Renders a sleek Radar / Spider chart showing candidate skill density across domains."""
    if not categories or not candidate_skills:
        return None

    candidate_skills_lower = set(str(s).lower() for s in candidate_skills)
    domain_names = []
    coverage_pcts = []

    for domain, domain_skills in categories.items():
        if isinstance(domain_skills, list) and len(domain_skills) > 0:
            matched = sum(1 for s in domain_skills if str(s).lower() in candidate_skills_lower)
            if matched > 0:
                pct = min(100.0, round((matched / min(8, len(domain_skills))) * 100, 1))
                domain_names.append(domain.replace('_', ' ').title())
                coverage_pcts.append(pct)

    if len(domain_names) < 3:
        return None

    radar_labels = domain_names + [domain_names[0]]
    radar_vals = coverage_pcts + [coverage_pcts[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_vals,
        theta=radar_labels,
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.20)',
        line=dict(color='#2563eb', width=2),
        marker=dict(size=6, color='#1d4ed8')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#e2e8f0'),
            angularaxis=dict(tickfont=dict(size=11, color='#334155'))
        ),
        showlegend=False,
        margin=dict(l=30, r=30, t=25, b=25),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def render_match_signals_chart(skill_score: float, semantic_score: float, exp_score: float) -> go.Figure:
    """Grouped bar visualizer showing individual signal contributions."""
    signals = ["Skill Overlap (50%)", "Semantic Vector (40%)", "Tenure Alignment (10%)"]
    vals = [round(skill_score * 100, 1), round(semantic_score * 100, 1), round(exp_score * 100, 1)]
    colors = ["#10b981", "#2563eb", "#8b5cf6"]

    fig = go.Figure(go.Bar(
        x=signals,
        y=vals,
        marker=dict(color=colors),
        text=[f"{v}%" for v in vals],
        textposition='auto',
        textfont=dict(size=12, color='#ffffff')
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=15, b=10),
        height=220,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor='#f1f5f9', ticksuffix='%')
    )
    return fig
