import streamlit as st
import pandas as pd
import google.generativeai as genai
import ast
import re
import os

# --- AUTO-CONFIGURATION (The Branding Fix) ---
def setup_branding_config():
    config_dir = ".streamlit"
    config_path = os.path.join(config_dir, "config.toml")
    
    toml_content = """
[theme]
primaryColor="#ffbe3c"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#31333F"
font="sans serif"
"""
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    with open(config_path, "w") as f:
        f.write(toml_content)

setup_branding_config()

# --- Page Configuration ---
st.set_page_config(
    page_title="Sinch Connector Tool",
    page_icon="🔗",
    layout="centered"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    /* GLOBAL FONT COLORS */
    h1, h2, h3, .stMarkdown {
        color: #007171 !important; /* Sinch Dark Green */
    }
    
    /* BUTTON STYLING (Outline Style) */
    [data-testid="stForm"] button {
        background-color: #ffffff !important; 
        border: 2px solid #ffbe3c !important; /* Sinch Yellow Border */
        color: #31333F !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stForm"] button:hover {
        background-color: #ffbe3c !important; /* Sinch Yellow Fill */
        color: #000000 !important;
        border: 2px solid #ffbe3c !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    [data-testid="stForm"] button:active {
        background-color: #ffcf6e !important;
        transform: translateY(2px) !important;
    }

    /* COMPACT RESULT CARD */
    .employee-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 6px solid #ffbe3c;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    .emp-header {
        display: flex;
        align-items: baseline;
        gap: 10px;
        margin-bottom: 5px;
    }
    
    .emp-name {
        color: #007171;
        font-size: 1.2rem;
        font-weight: 800;
    }
    
    .emp-role {
        color: #666;
        font-size: 0.95rem;
        font-weight: 500;
        font-style: italic;
    }
    
    .emp-bio {
        color: #333;
        font-size: 0.9rem;
        line-height: 1.4;
        margin-bottom: 8px;
    }
    
    .emp-email a {
        color: #3aa7ea;
        text-decoration: none;
        font-size: 0.85rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        for col in ['Job Title', 'Bio', 'Skills', 'Expertise', 'Email', 'Name', 'Department']:
            df[col] = df[col].fillna('').astype(str)
        return df
    except FileNotFoundError:
        return None

# --- AI SETUP ---
@st.cache_resource
def get_model(api_key):
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        model_name = next((m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods), 'models/gemini-pro')
        return genai.GenerativeModel(model_name)
    except:
        return None

# --- SEARCH LOGIC (Smart Hybrid) ---
def search_logic(df, query, model, department_filter):
    # 1. Filter
    if department_filter != "All Departments":
        df = df[df['Department'] == department_filter]
    
    if df.empty: return [], "No matches in this department."

    # 2. Code Search (Instant)
    # Aggressive Stop Words to remove "Corporate Noise"
    stop_words = {
        'who', 'can', 'help', 'with', 'questions', 'about', 'find', 'me', 'a', 'an', 'the', 
        'i', 'have', 'question', 'whether', 'sinch', 'offers', 'solution', 'solutions', 
        'compliant', 'compliance', 'looking', 'need', 'know', 'expert'
    }
    keywords = [w for w in re.split(r'\W+', query.lower()) if w and w not in stop_words]

    if not keywords:
        return df.head(10).index.tolist(), "Showing recent employees."

    df = df.copy()
    df['score'] = 0
    for kw in keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        # UPDATED SCORING WEIGHTS per your request
        df['score'] += df['Job Title'].str.contains(pattern, case=False, regex=True).astype(int) * 10
        df['score'] += df['Skills'].str.contains(pattern, case=False, regex=True).astype(int) * 10
        df['score'] += df['Expertise'].str.contains(pattern, case=False, regex=True).astype(int) * 5
        df['score'] += df['Bio'].str.contains(pattern, case=False, regex=True).astype(int) * 3

    # Send top 40 to AI to ensure we catch the relevant people
    top_candidates = df[df['score'] > 0].sort_values('score', ascending=False).head(40)
    
    if top_candidates.empty: return [], "No direct keyword matches found."

    # 3. AI Re-ranking
    if model and len(top_candidates) > 3:
        try:
            prompt = f"""
            Act as a recruiter. Select the top 3-5 candidates from this list who best match the user's intent: "{query}".
            If specific technical terms (like HIPAA, Python, Marketo) are used, PRIORITIZE candidates who have those exact skills over generalists.
            Return ONLY a Python list of indices (e.g. [5, 12]).
            List:
            {top_candidates[['Name', 'Job Title', 'Bio', 'Skills']].to_csv()}
            """
            response = model.generate_content(prompt)
            indices = [int(n) for n in re.findall(r'\d+', response.text)]
            valid_indices = [i for i in indices if i in top_candidates.index]
            if valid_indices: return valid_indices, None
        except:
            pass 

    return top_candidates.head(5).index.tolist(), None

# --- UI LAYOUT ---

with st.sidebar:
    st.header("⚙️ Config")
    api_key = st.text_input("Google AI API Key", type="password")
    st.markdown("---")
    st.caption("Without a key, the tool uses Keyword Match only.")

# Header
left_co, cent_co, last_co = st.columns([3, 2, 3])
with cent_co:
    if os.path.exists("sinch_logo.png"):
        st.image("sinch_logo.png", use_container_width=True)
    elif os.path.exists("sinch_logo.jpg"):
        st.image("sinch_logo.jpg", use_container_width=True)

st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>Sinch Connector Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Find colleagues fast. Select a department or just search.</p>", unsafe_allow_html=True)

# Load Data
df = load_data('sinch_directory.xlsx')

if df is not None:
    departments = ["All Departments"] + sorted(df['Department'].unique().tolist())
    
    dept_filter = st.selectbox("Filter by Department:", departments)
    
    with st.form(key='search_form'):
        query = st.text_input("Search", placeholder="e.g. WhatsApp pricing expert")
        submit = st.form_submit_button("Find Sinchers")

    if submit and query:
        model = get_model(api_key)
        with st.spinner("Searching..."):
            results, error = search_logic(df, query, model, dept_filter)
        
        if results:
            st.markdown("---")
            for i in results:
                row = df.loc[i]
                st.markdown(f"""
                <div class="employee-card">
                    <div class="emp-header">
                        <span class="emp-name">{row['Name']}</span>
                        <span class="emp-role">{row['Job Title']}</span>
                    </div>
                    <div class="emp-bio">{row['Bio']}</div>
                    <div class="emp-email">
                        <a href="mailto:{row['Email']}">✉️ Email {row['Name'].split()[0]}</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        elif error:
            st.warning(error)
        else:
            st.info("No results found.")
else:
    st.error("File not found.")
