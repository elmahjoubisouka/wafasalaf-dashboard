"""
Wafasalaf — Dashboard Veille Publicitaire IA
Version corrigée : bouton Envoyer jaune + chat fonctionnel
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, warnings, datetime, io, os
import requests
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import cm

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Wafasalaf · Veille IA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
DRIVE_CSV_ID  = st.secrets.get("DRIVE_CSV_ID",  "")
DRIVE_JSON_ID = st.secrets.get("DRIVE_JSON_ID", "")
MODEL_NAME    = "llama-3.3-70b-versatile"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Open+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif !important;
    background-color: #1A1A1A !important;
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 3px solid #F5C400 !important;
}
[data-testid="stSidebar"] * { color: #FFFFFF !important; }

.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #1A1A1A !important;
}
[data-testid="stMainBlockContainer"] {
    background-color: #1A1A1A !important;
}

[data-testid="metric-container"] {
    background: #2C2C2C;
    border: 1px solid rgba(245,196,0,0.25);
    border-top: 3px solid #F5C400;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 10px !important;
    color: #F5C400 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #00B5A0 !important;
}

[data-testid="stTabs"] [role="tablist"] {
    background: #2C2C2C;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(245,196,0,0.15);
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 7px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    color: #aaaaaa !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #F5C400 !important;
    color: #1A1A1A !important;
    font-weight: 700 !important;
    border: none !important;
}

/* ── TOUS LES BOUTONS EN JAUNE (y compris form submit) ── */
.stButton > button,
[data-testid="stFormSubmitButton"] > button,
button[kind="primary"],
button[kind="secondaryFormSubmit"] {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    background: #F5C400 !important;
    color: #1A1A1A !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background: #e6b800 !important;
    color: #1A1A1A !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(245,196,0,0.3) !important;
}

/* Bouton Effacer en gris */
.stButton > button[kind="secondary"] {
    background: #2C2C2C !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(245,196,0,0.3) !important;
}

.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: #2C2C2C !important;
    border: 1px solid rgba(245,196,0,0.3) !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #F5C400 !important;
    box-shadow: 0 0 0 2px rgba(245,196,0,0.15) !important;
}

/* ── CHAT ── */
.chat-user {
    background: linear-gradient(135deg, #F5C400, #e6a800);
    color: #1A1A1A;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 6px 0 6px 60px;
    font-size: 14px;
    line-height: 1.65;
    font-weight: 500;
}
.chat-ai {
    background: #2C2C2C;
    border: 1px solid rgba(0,181,160,0.2);
    border-left: 3px solid #00B5A0;
    color: #FFFFFF;
    border-radius: 4px 16px 16px 16px;
    padding: 12px 16px;
    margin: 6px 60px 6px 0;
    font-size: 14px;
    line-height: 1.7;
}
.chat-meta {
    font-size: 10px;
    color: #888888;
    font-family: 'Montserrat', monospace;
    margin-bottom: 3px;
}

.pred-card {
    background: #2C2C2C;
    border: 1px solid rgba(245,196,0,0.1);
    border-left: 4px solid #F5C400;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
}
.pred-card.high  { border-left-color: #e74c3c; }
.pred-card.medium { border-left-color: #F5C400; }
.pred-card.low   { border-left-color: #00B5A0; }

.section-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    margin-bottom: 2px;
    border-left: 4px solid #F5C400;
    padding-left: 12px;
}
.section-sub {
    font-size: 13px;
    color: #888888;
    margin-bottom: 18px;
    padding-left: 16px;
}

.status-ok {
    background: rgba(0,181,160,0.1);
    border: 1px solid rgba(0,181,160,0.35);
    color: #00B5A0;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12px;
    font-family: 'Montserrat', monospace;
}
.status-err {
    background: rgba(231,76,60,0.1);
    border: 1px solid rgba(231,76,60,0.35);
    color: #e74c3c;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12px;
    font-family: 'Montserrat', monospace;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #2C2C2C !important;
    border: 1px solid rgba(245,196,0,0.15) !important;
    border-radius: 12px !important;
}

[data-testid="stDataFrame"] {
    background: #2C2C2C !important;
    border-radius: 10px !important;
    border: 1px solid rgba(245,196,0,0.15) !important;
}

hr { border-color: rgba(245,196,0,0.2) !important; }

[data-testid="stSidebar"] .stRadio label {
    color: #CCCCCC !important;
    font-size: 14px !important;
    padding: 4px 0 !important;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #F5C400 !important; }

#MainMenu, footer, header { visibility: hidden; }

.stProgress > div > div { background-color: #F5C400 !important; }

[data-testid="stExpander"] {
    background: #2C2C2C !important;
    border: 1px solid rgba(245,196,0,0.15) !important;
    border-radius: 10px !important;
}

.stSpinner > div { border-top-color: #F5C400 !important; }

.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

MARQUES = ['Wafasalaf', 'Salafin', 'Eqdom', 'Sofac']
COLORS  = {'Wafasalaf':'#F5C400','Salafin':'#00B5A0','Eqdom':'#FF6B35','Sofac':'#8B5CF6'}
BG2='#1A1A1A'; BG3='#2C2C2C'; GRID='rgba(245,196,0,0.07)'; TEXT='#FFFFFF'; TEXT2='#888888'
PLOT_L = dict(
    paper_bgcolor=BG2, plot_bgcolor=BG3,
    font=dict(family='Open Sans', color=TEXT2, size=12),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor=GRID, zeroline=False, color=TEXT2),
    yaxis=dict(gridcolor=GRID, zeroline=False, color=TEXT2),
)

for k, v in [
    ('client', None), ('df', None), ('context_json', None),
    ('historique_chat', []), ('last_rapport', ''),
    ('last_load', None), ('load_status', ''),
]:
    if k not in st.session_state:
        st.session_state[k] = v

def download_from_drive(file_id: str, suffix: str) -> str:
    import gdown
    out = f"/tmp/wafa_{file_id[:8]}{suffix}"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, out, quiet=True)
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def load_from_drive(csv_id: str, json_id: str):
    try:
        csv_path  = download_from_drive(csv_id,  ".csv")
        json_path = download_from_drive(json_id, ".json")
        df_raw = pd.read_csv(csv_path, low_memory=False)
        df_raw['post_date'] = pd.to_datetime(df_raw['post_date'], utc=True, errors='coerce')
        df_raw['post_date'] = df_raw['post_date'].dt.tz_localize(None)
        df_raw = df_raw.rename(columns={
            'company':'marque',        'post_date':'date_debut',
            'post_text':'texte_complet','comments_count':'comments',
            'Accroche_pub':'accroche', 'Type_offre':'type_offre',
            'Langue':'langue',         'Montant_finance':'montant',
            'Objet_finance':'objet_finance',
        })
        df_raw['engagement'] = (
            df_raw.get('likes',     pd.Series(0, index=df_raw.index)).fillna(0) * 1 +
            df_raw.get('reactions', pd.Series(0, index=df_raw.index)).fillna(0) * 1 +
            df_raw.get('comments',  pd.Series(0, index=df_raw.index)).fillna(0) * 2 +
            df_raw.get('shares',    pd.Series(0, index=df_raw.index)).fillna(0) * 3
        )
        with open(json_path, encoding='utf-8') as f:
            ctx = json.load(f)
        return df_raw, ctx, None
    except Exception as e:
        return None, None, str(e)

def auto_load():
    if st.session_state.df is not None:
        return
    if not DRIVE_CSV_ID or not DRIVE_JSON_ID:
        st.session_state.load_status = "no_ids"
        return
    with st.spinner("Chargement automatique depuis Google Drive..."):
        df, ctx, err = load_from_drive(DRIVE_CSV_ID, DRIVE_JSON_ID)
    if err:
        st.session_state.load_status = f"error:{err}"
    else:
        st.session_state.df           = df
        st.session_state.context_json = ctx
        st.session_state.last_load    = datetime.datetime.now()
        st.session_state.load_status  = "ok"

auto_load()

def need_data():
    if st.session_state.df is None:
        st.info("Les données se chargent automatiquement depuis Google Drive.\n\n"
                "Si rien ne s'affiche, allez dans Configuration.")
        st.stop()

def need_api():
    if not st.session_state.client:
        st.warning("Clé API Groq non configurée. Allez dans Configuration.")
        st.stop()

def is_empty(val):
    if pd.isna(val): return True
    return str(val).strip() in ['', '—', '--', 'nan']

def init_groq():
    if GROQ_API_KEY and st.session_state.client is None:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            st.session_state.client = client
        except Exception:
            pass

init_groq()

def llm_generate(prompt: str) -> str:
    if not st.session_state.client:
        st.error("Clé API Groq non configurée. Allez dans Configuration.")
        return ''
    try:
        response = st.session_state.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erreur API Groq : {e}")
        return ''

def construire_contexte_rag(df, ctx, marque_focus='Wafasalaf',
                             date_debut=None, date_fin=None, max_posts=25):
    concurrents = [m for m in MARQUES if m != marque_focus]
    dff = df[df['marque'].isin([marque_focus] + concurrents)].copy()
    if date_debut: dff = dff[dff['date_debut'] >= pd.to_datetime(date_debut)]
    if date_fin:   dff = dff[dff['date_debut'] <= pd.to_datetime(date_fin)]
    dei = ctx.get('classement_DEI', {})
    c  = "BENCHMARK DEI - CREDIT CONSOMMATION MAROC\n"
    c += f"Periode : {ctx['metadata']['periode_analyse']}\n"
    c += f"Focus : {marque_focus}\n\nCLASSEMENT DEI (/100) :\n"
    for m, sc in sorted(dei.get('classement', {}).items(), key=lambda x: -x[1]):
        c += f"  {m} : {sc}/100{'  <- FOCUS' if m == marque_focus else ''}\n"
    c += f"  Moyenne : {dei.get('moyenne_marche')}\n\n"
    for marque in [marque_focus] + concurrents:
        info = ctx.get('analyse_par_marque', {}).get(marque, {})
        if not info: continue
        eng  = info.get('engagement', {})
        edit = info.get('strategie_editoriale', {})
        ind  = info.get('DEI_indicateurs_normalises', {})
        c += f"--- {marque.upper()} ---\n"
        c += f"DEI : {info.get('DEI_score')}/100 (rang #{info.get('DEI_rang')})\n"
        c += f"Eng median : {eng.get('median')} | Commercial : {eng.get('commercial_median')}\n"
        c += f"Langue dominante : {edit.get('langue_dominante')} | Offre top : {edit.get('offre_plus_performante')}\n"
        c += f"Frequence={ind.get('Frequence')} | Diversite={ind.get('Diversite')} | Richesse={ind.get('Richesse')}\n\n"
    bench = ctx.get('benchmark_engagement', {})
    el    = bench.get('engagement_par_langue', {}).get('median', {})
    c += f"LANGUE : Arabe={el.get('Arabe')} | Darija={el.get('Darija')} | Francais={el.get('Français', el.get('Francais'))}\n"
    c += f"Correlation DEI-engagement : r={bench.get('correlation_DEI_engagement')}\n\n"
    cols = [col for col in ['marque','date_debut','texte_complet','type_offre','langue','engagement'] if col in dff.columns]
    for _, row in dff[cols].dropna(subset=['texte_complet']).head(max_posts).iterrows():
        c += f"[{row['marque']}|{str(row['date_debut'])[:10]}] {str(row.get('texte_complet',''))[:100]}\n"
    return c

def chat_llm(question, date_debut=None, date_fin=None):
    if not st.session_state.client:
        st.error("Clé API Groq non configurée.")
        return ''
    df  = st.session_state.df
    ctx = st.session_state.context_json
    rag = construire_contexte_rag(df, ctx, date_debut=date_debut, date_fin=date_fin)
    system_msg = ('Tu es assistant strategique marketing Wafasalaf au Maroc. '
                  'Reponds en francais professionnel avec des chiffres precis. '
                  "Tu comprends et analyses aussi le Darija et l'arabe marocain.\n\nDONNEES :\n" + rag[:5000])
    messages = [{"role": "system", "content": system_msg}]
    for t in st.session_state.historique_chat[-6:]:
        messages.append({"role": "user",      "content": t['question']})
        messages.append({"role": "assistant", "content": t['reponse']})
    messages.append({"role": "user", "content": question})
    try:
        response = st.session_state.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
        )
        rep = response.choices[0].message.content
    except Exception as e:
        st.error(f"Erreur API Groq : {e}")
        return ''
    st.session_state.historique_chat.append({
        'question': question, 'reponse': rep,
        'timestamp': datetime.datetime.now().isoformat()
    })
    return rep

def comparer_periode(date_debut, date_fin, focus='Wafasalaf'):
    df = st.session_state.df
    d1 = pd.to_datetime(date_debut); d2 = pd.to_datetime(date_fin)
    dff = df[(df['date_debut'] >= d1) & (df['date_debut'] <= d2)].copy()
    if len(dff) == 0:
        return None, None, f'Aucun post entre {date_debut} et {date_fin}'
    stats = {}
    for m in MARQUES:
        dm = dff[dff['marque'] == m]
        if len(dm) == 0: continue
        nb_j = max(1, (d2 - d1).days)
        top  = ''
        if 'accroche' in dm.columns and dm['engagement'].max() > 0:
            top = str(dm.loc[dm['engagement'].idxmax(), 'accroche'])[:80]
        stats[m] = {
            'nb_posts': len(dm),
            'posts_semaine': round(len(dm) / (nb_j / 7), 1),
            'engagement_median': int(dm['engagement'].median()),
            'engagement_max': int(dm['engagement'].max()),
            'langues': dm['langue'].value_counts().to_dict(),
            'top_accroche': top,
        }
    prompt = (f'Consultant marketing Wafasalaf Maroc.\nPeriode : {date_debut} -> {date_fin}\n'
              f'Stats : {json.dumps(stats, ensure_ascii=False)}\n\n'
              f'Analyse focalisee {focus} :\n'
              f'1. SYNTHESE (3 phrases, chiffres)\n'
              f'2. POSITION {focus.upper()} vs concurrents\n'
              f'3. CONCURRENT LE PLUS MENACANT\n'
              f'4. 3 RECOMMANDATIONS avec delai')
    return stats, llm_generate(prompt), None

def predire_prochaines_offres(concurrent, horizon='3 prochains mois', nb=4):
    df  = st.session_state.df
    ctx = st.session_state.context_json
    dc  = df[df['marque'] == concurrent].copy()
    if len(dc) == 0: return None
    saisons = {1:'Debut annee',2:'St-Valentin',3:'Printemps',4:'Ramadan',5:'Aid',
               6:'Ete',7:'Vacances',8:'Rentree prep',9:'Rentree',
               10:'Automne',11:'Black Friday',12:'Fetes'}
    info   = ctx.get('analyse_par_marque', {}).get(concurrent, {})
    prompt = (f'Expert marketing credit consommation Maroc.\n'
              f'Predit {nb} prochaines campagnes de {concurrent} sur {horizon}.\n\n'
              f'DONNEES {concurrent.upper()} :\n'
              f'- Posts : {len(dc)}\n'
              f'- Offres : {dc["type_offre"].value_counts().to_dict()}\n'
              f'- Langues : {dc["langue"].value_counts().to_dict()}\n'
              f'- Eng median : {round(dc["engagement"].median(),0)} | Max : {int(dc["engagement"].max())}\n'
              f'- DEI : {info.get("DEI_score","?")}/100\n'
              f'- Saison actuelle : {saisons.get(datetime.datetime.now().month,"Standard")}\n\n'
              'Reponds UNIQUEMENT JSON valide sans markdown :\n'
              '{"predictions":[{"rang":1,"type_offre":"...","titre_probable":"...",'
              '"langue_probable":"Darija|Francais|Arabe","montant_probable":"...",'
              '"canal_probable":"Facebook|Instagram","timing_probable":"...",'
              '"probabilite":"Haute|Moyenne|Faible","justification":"...",'
              '"contre_mesure_wafasalaf":"..."}],'
              '"risque_global":"...","opportunite_pour_wafasalaf":"..."}')
    raw = llm_generate(prompt).strip()
    try:
        if '```' in raw:
            raw = raw.split('```')[1]
            if raw.startswith('json'): raw = raw[4:]
            raw = raw.split('```')[0].strip()
        return json.loads(raw)
    except Exception as e:
        return {'predictions': [], 'erreur': str(e), 'texte_brut': raw}

def generer_rapport_pdf(date_debut=None, date_fin=None):
    df  = st.session_state.df
    ctx = st.session_state.context_json
    rag = construire_contexte_rag(df, ctx, max_posts=15)
    periode = f'{date_debut} -> {date_fin}' if date_debut else ctx['metadata']['periode_analyse']
    prompt = (
        'Consultant senior marketing digital Wafasalaf (credit consommation, Maroc).\n\n'
        + rag[:4000] +
        '\n\nRedige un rapport strategique avec ces sections :\n\n'
        'RESUME EXECUTIF\n4 phrases sur la situation concurrentielle.\n\n'
        'ANALYSE DE LA POSITION CONCURRENTIELLE\nCompare avec chiffres.\n\n'
        'OPPORTUNITES IDENTIFIEES\n3 opportunites concretes.\n\n'
        'RECOMMANDATIONS STRATEGIQUES\n5 actions avec impact et delai.\n\n'
        'CONCLUSION\nPositionnement recommande 6 mois.\n'
    )
    contenu = llm_generate(prompt)
    if not contenu: return None, ''
    buf   = io.BytesIO()
    BLEU  = HexColor('#0055A5')
    GRIS  = HexColor('#F5F5F5')
    TEXTE = HexColor('#444444')
    doc   = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    s_h1  = ParagraphStyle('h1', fontSize=20, textColor=BLEU, fontName='Helvetica-Bold', spaceAfter=6)
    s_h2  = ParagraphStyle('h2', fontSize=13, textColor=BLEU, fontName='Helvetica-Bold', spaceBefore=12, spaceAfter=4)
    s_bod = ParagraphStyle('bod', fontSize=10, textColor=TEXTE, leading=16, spaceAfter=5)
    s_met = ParagraphStyle('met', fontSize=9,  textColor=TEXTE, fontName='Helvetica-Oblique', spaceAfter=10)
    story = [
        Paragraph('Rapport de Veille Publicitaire Digitale', s_h1),
        Paragraph('Wafasalaf — Benchmark Concurrentiel', s_h2),
        Paragraph(f"Periode : {periode}", s_met),
        Spacer(1, 0.4*cm),
        Paragraph('Classement DEI', s_h2),
    ]
    rows = [['Marque', 'Score /100', 'Rang', 'vs Moy.']]
    moy  = ctx['classement_DEI']['moyenne_marche']
    for i, (m, sc) in enumerate(sorted(ctx['classement_DEI']['classement'].items(), key=lambda x: -x[1]), 1):
        e = round(sc - moy, 1)
        rows.append([m, str(sc), f'#{i}', f"{'+' if e >= 0 else ''}{e}"])
    t = Table(rows, colWidths=[4.5*cm, 3*cm, 2.5*cm, 4*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), BLEU), ('TEXTCOLOR', (0,0),(-1,0), white),
        ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9),
        ('ALIGN',      (1,0),(-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [GRIS, HexColor('#FFFFFF')]),
        ('GRID',       (0,0),(-1,-1), 0.25, HexColor('#CCCCCC')),
        ('TOPPADDING', (0,0),(-1,-1), 5), ('BOTTOMPADDING',(0,0),(-1,-1), 5),
    ]))
    story += [t, Spacer(1, 0.5*cm), Paragraph('Analyse Strategique', s_h2)]
    TITRES = ['RESUME EXECUTIF','ANALYSE DE LA POSITION CONCURRENTIELLE',
              'OPPORTUNITES IDENTIFIEES','RECOMMANDATIONS STRATEGIQUES','CONCLUSION']
    for line in contenu.split('\n'):
        line = line.strip()
        if not line: story.append(Spacer(1, 0.2*cm)); continue
        if any(line.upper().startswith(t) for t in TITRES):
            story.append(Paragraph(line, s_h2))
        else:
            story.append(Paragraph(
                line.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'), s_bod))
    doc.build(story)
    return buf.getvalue(), contenu

# ── GRAPHIQUES ────────────────────────────────────────────────────
def chart_dei(ctx):
    dei  = ctx['classement_DEI']['classement']
    moy  = ctx['classement_DEI']['moyenne_marche']
    data = sorted(dei.items(), key=lambda x: x[1])
    fig  = go.Figure(go.Bar(
        x=[v for _,v in data], y=[m for m,_ in data], orientation='h',
        marker_color=[COLORS[m] for m,_ in data],
        text=[str(v) for _,v in data], textposition='outside',
    ))
    fig.add_vline(x=moy, line=dict(color=TEXT2, dash='dash'),
                  annotation_text=f'Moy. {moy}', annotation_font_color=TEXT2)
    fig.update_layout(
        paper_bgcolor=BG2, plot_bgcolor=BG3,
        font=dict(family='Open Sans', color=TEXT2, size=12),
        margin=dict(l=16, r=16, t=40, b=16),
        height=260, showlegend=False,
        title=dict(text='Classement DEI Global', font=dict(size=14, color=TEXT)),
        xaxis=dict(range=[0,110], gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False),
    )
    return fig

def chart_radar(ctx):
    labels = ['Frequence','Diversite','Mix Langue','Richesse','Regularite']
    keys   = ['Frequence','Diversite','Mix_langue','Richesse','Regularite']
    fig    = go.Figure()
    for m in MARQUES:
        ind  = ctx['analyse_par_marque'].get(m, {}).get('DEI_indicateurs_normalises', {})
        vals = [ind.get(k, ind.get(k.lower(), 50)) for k in keys]
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=labels+[labels[0]],
            fill='toself', name=m, line=dict(color=COLORS[m], width=2),
        ))
    fig.update_layout(
        paper_bgcolor=BG2,
        polar=dict(bgcolor=BG3,
                   radialaxis=dict(visible=True, range=[0,100], gridcolor=GRID,
                                   tickfont=dict(size=9, color=TEXT2)),
                   angularaxis=dict(gridcolor=GRID, tickfont=dict(size=10, color=TEXT2))),
        font=dict(family='Open Sans', color=TEXT2),
        margin=dict(l=40,r=40,t=50,b=30), height=360,
        title=dict(text='Profil DEI - 5 Indicateurs', font=dict(size=14, color=TEXT)),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
    )
    return fig

def chart_engagement(df):
    eng = df.groupby('marque')['engagement'].agg(['median','mean']).reindex(MARQUES)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Mediane', x=MARQUES, y=eng['median'],
                         marker_color=[COLORS[m] for m in MARQUES],
                         text=eng['median'].round(0).astype(int), textposition='outside'))
    fig.add_trace(go.Bar(name='Moyenne', x=MARQUES, y=eng['mean'],
                         marker_color=[COLORS[m] for m in MARQUES], opacity=0.4))
    fig.update_layout(**PLOT_L, barmode='overlay', height=290,
                      title=dict(text='Engagement par Marque', font=dict(size=14, color=TEXT)))
    return fig

def chart_langue(df):
    df2   = df[~df['langue'].apply(is_empty)]
    stats = df2.groupby('langue')['engagement'].agg(['median','count']).reset_index()
    stats = stats.sort_values('median', ascending=False)
    LC    = {'Darija':'#4FC18A','Français':'#4F8EF7','Arabe':'#F7C44F'}
    fig   = go.Figure(go.Bar(
        x=stats['langue'], y=stats['median'],
        marker_color=[LC.get(l,'#888') for l in stats['langue']],
        text=[f"{v:.0f} (n={c})" for v,c in zip(stats['median'], stats['count'])],
        textposition='outside',
    ))
    fig.update_layout(**PLOT_L, height=280, showlegend=False,
                      title=dict(text='Engagement par Langue', font=dict(size=14, color=TEXT)))
    return fig

def chart_com_noncom(df):
    mask = df['type_offre'].apply(is_empty) & df['accroche'].apply(is_empty)
    mc   = [df[~mask & (df['marque']==m)]['engagement'].median() for m in MARQUES]
    mnc  = [df[ mask & (df['marque']==m)]['engagement'].median() for m in MARQUES]
    fig  = go.Figure()
    fig.add_trace(go.Bar(name='Commercial',     x=MARQUES, y=mc,
                         marker_color=[COLORS[m] for m in MARQUES]))
    fig.add_trace(go.Bar(name='Non-commercial', x=MARQUES, y=mnc,
                         marker_color=[COLORS[m] for m in MARQUES], opacity=0.35))
    fig.update_layout(**PLOT_L, barmode='group', height=290,
                      title=dict(text='Commercial vs Non-Commercial', font=dict(size=14, color=TEXT)))
    return fig

def chart_offre(df):
    df2   = df[~df['type_offre'].apply(is_empty)]
    stats = df2.groupby('type_offre')['engagement'].agg(['median','count']).reset_index()
    stats = stats.sort_values('median', ascending=True)
    fig   = go.Figure(go.Bar(
        y=[o.replace('Credit ','Cdt ') for o in stats['type_offre']],
        x=stats['median'], orientation='h',
        marker_color=['#4F8EF7','#4FC18A','#F7C44F'],
        text=[f"{v:.0f} (n={c})" for v,c in zip(stats['median'], stats['count'])],
        textposition='outside',
    ))
    fig.update_layout(**PLOT_L, height=240, showlegend=False,
                      title=dict(text="Engagement par Type d'Offre", font=dict(size=14, color=TEXT)))
    return fig

def chart_timeline(df):
    df2   = df.copy()
    df2['mois'] = df2['date_debut'].dt.to_period('M').astype(str)
    pivot = df2.groupby(['mois','marque']).size().reset_index(name='nb')
    fig   = go.Figure()
    for m in MARQUES:
        d = pivot[pivot['marque'] == m]
        fig.add_trace(go.Scatter(x=d['mois'], y=d['nb'], name=m,
                                  line=dict(color=COLORS[m], width=2),
                                  mode='lines+markers', marker=dict(size=5)))
    fig.update_layout(**PLOT_L, height=270, xaxis_tickangle=-35,
                      title=dict(text='Activite Mensuelle', font=dict(size=14, color=TEXT)))
    return fig

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid rgba(245,196,0,0.2);">
      <div style="width:40px;height:40px;border-radius:10px;background:#F5C400;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:900;color:#1A1A1A;">W</div>
      <div>
        <div style="font-family:'Montserrat',sans-serif;font-size:15px;font-weight:800;color:#FFFFFF;letter-spacing:-0.01em;">Wafasalaf</div>
        <div style="font-size:10px;color:#F5C400;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;">Veille Publicitaire IA</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📊 Dashboard", "💬 Chat LLM", "📅 Comparaison",
        "🔮 Prediction", "📄 Rapport PDF", "⚙️ Configuration",
    ], label_visibility='collapsed')

    st.divider()
    if st.session_state.df is not None:
        n  = len(st.session_state.df)
        ts = st.session_state.last_load
        lb = ts.strftime("Donnees du %d/%m a %H:%M") if ts else "Donnees chargees"
        st.markdown(f'<div class="status-ok">✓ {n} posts — {lb}</div>', unsafe_allow_html=True)
    elif st.session_state.load_status == "no_ids":
        st.markdown('<div class="status-err">IDs Drive non configures</div>', unsafe_allow_html=True)
    elif str(st.session_state.load_status).startswith("error"):
        st.markdown('<div class="status-err">Erreur chargement Drive</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Rafraichir les donnees", use_container_width=True):
        load_from_drive.clear()
        st.session_state.df           = None
        st.session_state.context_json = None
        st.rerun()

    if st.session_state.client:
        st.markdown('<div class="status-ok" style="margin-top:6px;">✓ Groq / Llama-3 connecte</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='font-size:10px;color:#5a5e75;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;'>DEI /100</div>", unsafe_allow_html=True)
    for m, sc in [('Salafin',71.2),('Wafasalaf',60.2),('Eqdom',58.4),('Sofac',13.6)]:
        c1, c2 = st.columns([3,1])
        c1.markdown(f"<span style='color:{COLORS[m]};font-size:12px;font-weight:500;'>{m}</span>", unsafe_allow_html=True)
        c1.progress(int(sc))
        c2.markdown(f"<span style='font-family:monospace;font-size:11px;color:{COLORS[m]};'>{sc}</span>", unsafe_allow_html=True)

# ── CONFIGURATION ─────────────────────────────────────────────────
if page == "⚙️ Configuration":
    st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Parametres Drive, API Groq (Llama-3) et upload manuel.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Google Drive — IDs des fichiers**")
        st.caption("Entre uniquement l'ID (pas le lien complet).")
        col1, col2 = st.columns(2)
        csv_id_input  = col1.text_input("ID du fichier CSV",  value=DRIVE_CSV_ID,
                                         placeholder="1aBcDeFgHiJkLmNoPqRsTuVwX")
        json_id_input = col2.text_input("ID du fichier JSON", value=DRIVE_JSON_ID,
                                         placeholder="1zYxWvUTsRqPoNmLkJiHgFeDcBa")
        if st.button("Sauvegarder et charger", type="primary", use_container_width=True):
            if csv_id_input and json_id_input:
                load_from_drive.clear()
                with st.spinner("Chargement depuis Drive..."):
                    df, ctx, err = load_from_drive(csv_id_input, json_id_input)
                if err:
                    st.error(f"Erreur : {err}")
                else:
                    st.session_state.df           = df
                    st.session_state.context_json = ctx
                    st.session_state.last_load    = datetime.datetime.now()
                    st.session_state.load_status  = "ok"
                    st.success(f"OK — {len(df)} posts charges depuis Drive !")
                    st.rerun()
            else:
                st.warning("Entre les deux IDs.")
        st.info("Lien Drive → copier la partie apres /d/ et avant /view")

    with st.container(border=True):
        st.markdown("**Cle API Groq (Llama 3.3)**")
        st.caption("Cle gratuite sur console.groq.com → API Keys")
        api_input = st.text_input("Cle API Groq", value=GROQ_API_KEY,
                                   type="password", label_visibility="collapsed",
                                   placeholder="gsk_...")
        if st.button("Connecter Groq", type="primary", use_container_width=True):
            try:
                test_client = Groq(api_key=api_input)
                test_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role":"user","content":"test"}],
                    max_tokens=5
                )
                st.session_state.client = test_client
                st.success("✓ Groq / Llama-3 connecte !")
            except Exception as e:
                st.error(f"Erreur Groq : {e}")

    with st.expander("Upload manuel (si Drive non configure)"):
        col1, col2 = st.columns(2)
        csv_file  = col1.file_uploader("social_posts_enrichi.csv", type=["csv"])
        json_file = col2.file_uploader("llm_context.json", type=["json"])
        if st.button("Charger les fichiers uploades", use_container_width=True):
            if csv_file and json_file:
                df_raw = pd.read_csv(csv_file, low_memory=False)
                df_raw['post_date'] = pd.to_datetime(df_raw['post_date'], utc=True, errors='coerce')
                df_raw['post_date'] = df_raw['post_date'].dt.tz_localize(None)
                df_raw = df_raw.rename(columns={
                    'company':'marque', 'post_date':'date_debut',
                    'post_text':'texte_complet', 'comments_count':'comments',
                    'Accroche_pub':'accroche', 'Type_offre':'type_offre',
                    'Langue':'langue', 'Montant_finance':'montant',
                    'Objet_finance':'objet_finance',
                })
                df_raw['engagement'] = (
                    df_raw.get('likes',     pd.Series(0, index=df_raw.index)).fillna(0) * 1 +
                    df_raw.get('reactions', pd.Series(0, index=df_raw.index)).fillna(0) * 1 +
                    df_raw.get('comments',  pd.Series(0, index=df_raw.index)).fillna(0) * 2 +
                    df_raw.get('shares',    pd.Series(0, index=df_raw.index)).fillna(0) * 3
                )
                st.session_state.df           = df_raw
                st.session_state.context_json = json.load(json_file)
                st.session_state.last_load    = datetime.datetime.now()
                st.success(f"OK — {len(df_raw)} posts charges !")
                st.rerun()
            else:
                st.warning("Uploade les deux fichiers.")

# ── DASHBOARD ─────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    need_data()
    df  = st.session_state.df
    ctx = st.session_state.context_json
    st.markdown('<div class="section-title">Dashboard — Benchmark DEI</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Vue comparative · 4 marques · 800 posts · Jan 2025 → Mar 2026</div>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Posts analyses",  "800",        "4 marques")
    k2.metric("Leader DEI",      "Salafin",    "71.2/100")
    k3.metric("Wafasalaf DEI",   "60.2/100",   "+9.4 vs moy.")
    k4.metric("Corr. DEI-Eng",   "r = 0.66",   "moderee")
    k5.metric("Langue top",      "Arabe/Darija","x26 vs FR")
    st.divider()
    c1,c2 = st.columns(2)
    c1.plotly_chart(chart_dei(ctx),        use_container_width=True)
    c2.plotly_chart(chart_radar(ctx),      use_container_width=True)
    c1,c2 = st.columns(2)
    c1.plotly_chart(chart_engagement(df),  use_container_width=True)
    c2.plotly_chart(chart_langue(df),      use_container_width=True)
    c1,c2 = st.columns(2)
    c1.plotly_chart(chart_com_noncom(df),  use_container_width=True)
    c2.plotly_chart(chart_offre(df),       use_container_width=True)
    st.plotly_chart(chart_timeline(df),    use_container_width=True)
    st.divider()
    st.markdown("#### Tableau recapitulatif")
    recap = []
    for m in MARQUES:
        info = ctx['analyse_par_marque'].get(m, {})
        ind  = info.get('DEI_indicateurs_normalises', {})
        eng  = info.get('engagement', {})
        recap.append({
            'Marque':      m,
            'DEI':         info.get('DEI_score', '-'),
            'Rang':        f"#{info.get('DEI_rang','-')}",
            'Frequence':   ind.get('Frequence', '-'),
            'Diversite':   ind.get('Diversite', '-'),
            'Richesse':    ind.get('Richesse', '-'),
            'Eng. median': eng.get('median', '-'),
        })
    st.dataframe(pd.DataFrame(recap), use_container_width=True, hide_index=True,
                 column_config={
                     'DEI': st.column_config.ProgressColumn('DEI', min_value=0, max_value=100, format='%.1f'),
                 })

# ── CHAT LLM ──────────────────────────────────────────────────────
elif page == "💬 Chat LLM":
    need_data()
    need_api()
    st.markdown('<div class="section-title">Chat Strategique</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Questions libres — contexte RAG injecte automatiquement.</div>', unsafe_allow_html=True)

    # Questions suggérées
    st.markdown("**Questions suggerees :**")
    qs = [
        "Pourquoi Salafin domine le DEI ? Que doit faire Wafasalaf ?",
        "Quelle langue genere le plus d'engagement et pourquoi ?",
        "Compare Wafasalaf et Eqdom sur la diversite des offres",
        "Quelles sont les 3 faiblesses principales de Wafasalaf ?",
        "Quelle offre est la plus performante sur le marche ?",
        "Analyse la correlation DEI-engagement. Quels enseignements ?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(qs):
        label = q[:44] + '...' if len(q) > 44 else q
        if cols[i % 3].button(label, key=f"q{i}", use_container_width=True):
            st.session_state['pending_q'] = q

    st.divider()

    # Historique des échanges
    for turn in st.session_state.historique_chat:
        ts = turn.get('timestamp', '')[:16].replace('T', ' ')
        st.markdown(f'<div class="chat-meta">Vous · {ts}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-user">{turn["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-meta">Assistant IA</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-ai">{turn["reponse"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── ZONE DE SAISIE SANS st.form ──────────────────────────────
    # On utilise on_change + session_state pour éviter le bug CSS du bouton rouge
    pending = st.session_state.pop("pending_q", "")

    user_q = st.text_area(
        "Votre question",
        height=90,
        placeholder="Posez votre question strategique...",
        value=pending,
        key="chat_input",
    )

    col_send, col_clear = st.columns([5, 1])

    with col_send:
        send_btn = st.button("Envoyer ✉", type="primary", use_container_width=True, key="btn_send")

    with col_clear:
        clear_btn = st.button("Effacer", use_container_width=True, key="btn_clear")

    if clear_btn:
        st.session_state.historique_chat = []
        st.rerun()

    if send_btn:
        question = st.session_state.get("chat_input", "").strip()
        if question:
            with st.spinner("Analyse en cours..."):
                chat_llm(question)
            st.rerun()
        else:
            st.warning("Veuillez saisir une question.")

    # Téléchargement historique
    if st.session_state.historique_chat:
        hist = "\n\n".join(
            f"Q: {t['question']}\nR: {t['reponse']}"
            for t in st.session_state.historique_chat
        )
        st.download_button(
            "Telecharger l'historique",
            data=hist.encode(),
            file_name="chat_wafasalaf.txt",
            mime="text/plain",
        )

    st.caption(f"{len(st.session_state.historique_chat)} echange(s)")

# ── COMPARAISON ───────────────────────────────────────────────────
elif page == "📅 Comparaison":
    need_data()
    need_api()
    st.markdown('<div class="section-title">Comparaison par Periode</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Analyse LLM + graphiques filtres sur une fenetre temporelle.</div>', unsafe_allow_html=True)

    if 'cmp_start' not in st.session_state: st.session_state.cmp_start = datetime.date(2025,6,1)
    if 'cmp_end'   not in st.session_state: st.session_state.cmp_end   = datetime.date(2025,12,31)

    with st.container(border=True):
        pc1,pc2,pc3,pc4 = st.columns(4)
        if pc1.button("S1 2025",  use_container_width=True):
            st.session_state.cmp_start, st.session_state.cmp_end = datetime.date(2025,1,1),  datetime.date(2025,6,30)
            st.rerun()
        if pc2.button("S2 2025",  use_container_width=True):
            st.session_state.cmp_start, st.session_state.cmp_end = datetime.date(2025,7,1),  datetime.date(2025,12,31)
            st.rerun()
        if pc3.button("Q1 2026",  use_container_width=True):
            st.session_state.cmp_start, st.session_state.cmp_end = datetime.date(2026,1,1),  datetime.date(2026,3,4)
            st.rerun()
        if pc4.button("Complete", use_container_width=True):
            st.session_state.cmp_start, st.session_state.cmp_end = datetime.date(2025,1,31), datetime.date(2026,3,4)
            st.rerun()

        c1,c2,c3 = st.columns(3)
        d_start  = c1.date_input("Date debut", value=st.session_state.cmp_start, key="cmp_d_start")
        d_end    = c2.date_input("Date fin",   value=st.session_state.cmp_end,   key="cmp_d_end")
        focus    = c3.selectbox("Focus", MARQUES)

        if d_start != st.session_state.cmp_start: st.session_state.cmp_start = d_start
        if d_end   != st.session_state.cmp_end:   st.session_state.cmp_end   = d_end

        run = st.button("Lancer l'analyse", type="primary", use_container_width=True)

    if run:
        with st.spinner("Analyse en cours..."):
            stats, analyse, err = comparer_periode(str(d_start), str(d_end), focus=focus)
        if err:
            st.warning(err)
        else:
            st.markdown(f"**{d_start} -> {d_end} | {sum(v['nb_posts'] for v in stats.values())} posts**")
            tbl = pd.DataFrame({m: {'Posts':v['nb_posts'],'Posts/sem':v['posts_semaine'],
                                    'Eng. median':v['engagement_median'],'Eng. max':v['engagement_max']}
                                for m,v in stats.items()}).T
            st.dataframe(tbl, use_container_width=True)
            df_f = st.session_state.df[
                (st.session_state.df['date_debut'] >= pd.Timestamp(d_start)) &
                (st.session_state.df['date_debut'] <= pd.Timestamp(d_end))
            ]
            c1,c2 = st.columns(2)
            c1.plotly_chart(chart_engagement(df_f), use_container_width=True)
            c2.plotly_chart(chart_langue(df_f),     use_container_width=True)
            with st.container(border=True):
                st.markdown(f"**Analyse LLM — {focus}**")
                st.markdown(analyse)
            st.download_button("Telecharger l'analyse", data=analyse.encode(),
                               file_name="comparaison.txt", mime="text/plain")

# ── PREDICTION ─────────────────────────────────────────────────────
elif page == "🔮 Prediction":
    need_data()
    need_api()
    st.markdown('<div class="section-title">Prediction des Prochaines Offres</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Anticipe les campagnes concurrentielles.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        c1,c2,c3 = st.columns(3)
        conc   = c1.selectbox("Concurrent", ['Salafin','Eqdom','Sofac'])
        horiz  = c2.selectbox("Horizon", ['1 prochain mois','3 prochains mois','6 prochains mois'], index=1)
        nb_p   = c3.selectbox("Nb predictions", [3,4,5], index=1)
        run_p  = st.button("Predire", type="primary", use_container_width=True)

    if run_p:
        with st.spinner(f"Analyse de {conc}..."):
            data = predire_prochaines_offres(conc, horiz, nb_p)
        if not data:
            st.error("Aucune donnee disponible.")
        elif data.get('predictions'):
            PCSS = {'Haute':'high','Moyenne':'medium','Faible':'low'}
            PEM  = {'Haute':'🔴 ROUGE','Moyenne':'🟡 ORANGE','Faible':'🟢 VERT'}
            st.markdown(f"#### Predictions — **{conc}** · {horiz}")
            for p in data['predictions']:
                prob = p.get('probabilite','Moyenne')
                st.markdown(f"""
                <div class="pred-card {PCSS.get(prob,'medium')}">
                  <div style="font-size:10px;color:#888;font-family:monospace;margin-bottom:5px;">
                    #{p.get('rang')} — {PEM.get(prob,'')} — Probabilite : {prob}
                  </div>
                  <div style="font-size:15px;font-weight:700;margin-bottom:3px;">{p.get('type_offre')}</div>
                  <div style="font-size:13px;color:#aaa;font-style:italic;margin-bottom:8px;">"{p.get('titre_probable')}"</div>
                  <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;">
                    <span style="font-size:10px;padding:2px 7px;border-radius:4px;background:#111;color:#888;">{p.get('langue_probable')}</span>
                    <span style="font-size:10px;padding:2px 7px;border-radius:4px;background:#111;color:#888;">{p.get('canal_probable')}</span>
                    <span style="font-size:10px;padding:2px 7px;border-radius:4px;background:#111;color:#888;">{p.get('timing_probable')}</span>
                    <span style="font-size:10px;padding:2px 7px;border-radius:4px;background:#111;color:#888;">{p.get('montant_probable','')}</span>
                  </div>
                  <div style="font-size:12px;color:#aaa;margin-bottom:7px;">{p.get('justification')}</div>
                  <div style="font-size:12px;padding:7px 11px;border-radius:7px;
                              background:rgba(245,196,0,0.08);border:1px solid rgba(245,196,0,0.2);color:#F5C400;">
                    → {p.get('contre_mesure_wafasalaf')}
                  </div>
                </div>""", unsafe_allow_html=True)
            st.divider()
            c1,c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("**Risque global**")
                    st.write(data.get('risque_global',''))
            with c2:
                with st.container(border=True):
                    st.markdown("**Opportunite Wafasalaf**")
                    st.write(data.get('opportunite_pour_wafasalaf',''))
        elif data.get('erreur'):
            st.error(f"Erreur JSON : {data['erreur']}")
            with st.expander("Reponse brute"):
                st.text(data.get('texte_brut',''))

# ── RAPPORT PDF ───────────────────────────────────────────────────
elif page == "📄 Rapport PDF":
    need_data()
    need_api()
    st.markdown('<div class="section-title">Rapport Strategique PDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Rapport complet genere par IA — exportable en PDF, TXT et HTML.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        c1,c2 = st.columns(2)
        r_start = c1.date_input("Debut", value=datetime.date(2025,1,1))
        r_end   = c2.date_input("Fin",   value=datetime.date(2026,3,4))
        gen_btn = st.button("Generer le rapport", type="primary", use_container_width=True)

    if gen_btn:
        with st.spinner("Generation en cours (30-60 sec)..."):
            pdf_bytes, contenu = generer_rapport_pdf(str(r_start), str(r_end))
        if pdf_bytes:
            st.session_state.last_rapport = contenu
            st.success("Rapport genere !")
            with st.container(border=True):
                st.markdown("**Apercu**")
                st.markdown(contenu)
            st.divider()
            ts   = datetime.date.today().strftime('%Y%m%d')
            name = f"rapport_wafasalaf_{ts}"
            c1,c2,c3 = st.columns(3)
            c1.download_button("Telecharger PDF",  data=pdf_bytes,
                               file_name=f"{name}.pdf", mime="application/pdf",
                               use_container_width=True, type="primary")
            c2.download_button("Telecharger TXT",  data=contenu.encode(),
                               file_name=f"{name}.txt", mime="text/plain",
                               use_container_width=True)
            html_out = (f"<!DOCTYPE html><html lang='fr'><head><meta charset='UTF-8'>"
                        f"<title>Rapport Wafasalaf</title>"
                        f"<style>body{{font-family:Georgia,serif;max-width:860px;margin:48px auto;"
                        f"padding:0 24px;line-height:1.85;color:#222;}}"
                        f"h1{{color:#0055A5;font-size:24px;border-bottom:3px solid #F5C400;"
                        f"padding-bottom:10px;}}h2{{color:#0055A5;font-size:16px;margin-top:28px;}}"
                        f"</style></head><body>"
                        f"<h1>Rapport Wafasalaf</h1>"
                        f"<p style='color:#888'>{r_start} -> {r_end}</p>"
                        f"{'<br>'.join(contenu.split(chr(10)))}</body></html>")
            c3.download_button("Telecharger HTML", data=html_out.encode(),
                               file_name=f"{name}.html", mime="text/html",
                               use_container_width=True)
    elif st.session_state.last_rapport:
        with st.expander("Dernier rapport genere"):
            st.markdown(st.session_state.last_rapport)
