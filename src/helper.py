import pandas as pd
from statistics import mean
import pandas as pd
import json
import numpy as np
from statistics import mean
import re
from datasets import load_dataset
import os
from collections import defaultdict
from src.envs import API, SAHARA_DATA, SAHARA_RESULTS
TASKS_LIST={
    'xlni':'Cross-Lingual Natural Language Inference',
    'lid':'Language Identification',
    'news': 'News Classification',
    'sentiment':'Sentiment Analysis',
    'topic':'Topic Classification',
    'mt_eng2xx':'Machine Translation - English to African',
    'mt_fra2xx':'Machine Translation - French to African',
    'mt_xx2xx':'Machine Translation - African to African',
    'paraphrase':'Paraphrase',
    'summary':'Summarization',
    'title':'Title Generation',
    'mmlu':'General Knowledge',
    'mgsm':'Mathematical Word Problems',
    'belebele':'Reading Comprehension',
    'squad_qa':'Context-based Question Answering',
    'ner':'Named Entity Recognition',
    'phrase':'Phrase Chunking',
    'pos':'Part-of-Speech Tagging',
}
CLUSTERS = {
    "Text Classification Tasks": [
        'xlni', 'lid', 'news', 'sentiment', 'topic',
    ],
    "Text Generation Tasks": [
        'mt_eng2xx', 'mt_fra2xx', 'mt_xx2xx', 'paraphrase', 'summary', 'title',
    ],
    "MCCR Tasks": [
        'mmlu', 'mgsm', 'belebele', 'squad_qa',
    ],
    "Tokens Level Tasks": [
        'ner', 'phrase', 'pos',
    ],
}
ALL_TASKS = [t for cluster in CLUSTERS.values() for t in cluster]
# This dictionary maps each task ID to its parent cluster name
TASK_TO_CLUSTER_MAP = {
    task: cluster_name
    for cluster_name, tasks in CLUSTERS.items()
    for task in tasks
}
# ===== Authenticate and Load Data From Private HF Repo =====

def load_private_leaderboard_df():
    ds = load_dataset(
        path=SAHARA_DATA,
        name=None,
        data_files=SAHARA_RESULTS,
        split="train",
        download_mode="force_redownload"
    )
    return ds.to_pandas()
metrics_list={
    'bleu_1k':'spBleu<sup>1K</sup>',
    'accuracy':'Accuracy',
    'f1':'Macro-F1',
    'exact_match':'Exact Match',
    'rougeL':'RougeL',
}
LANG_ISO2NAME = {
    'eng': 'English',
    'fra': 'French',
    # 'ara': 'Arabic',
    'amh': 'Amharic',
    'ewe': 'Ewe',
    'hau': 'Hausa',
    'ibo': 'Igbo',
    'kin': 'Kinyarwanda',
    'lin': 'Lingala',
    'lug': 'Ganda',
    'orm': 'Oromo',
    'sna': 'Shona',
    'sot': 'Southern Sotho',
    'swa': 'Swahili', 'swh': 'Swahili',
    'twi': 'Twi',
    'wol': 'Wolof',
    'xho': 'Xhosa',
    'yor': 'Yoruba',
    'zul': 'Zulu',
    'afr': 'Afrikaans',
    'run': 'Rundi',
    'tir': 'Tigrinya',
    'som': 'Somali',
    'pcm': 'Nigerian Pidgin',
    'teo': 'Teso',
    'nyn': 'Nyankore',# (Nyankole)',
    'lgg': 'Lugbara',
    'bem': 'Bemba',# (Chibemba)',
    'tsn': 'Tswana',
    'bbj': 'Ghomálá',
    'mos': 'Moore',
    'bam': 'Bambara',
    'fon': 'Fon',
    'ach': 'Acholi',
    'nso': 'Sepedi',
    'tso': 'Tsonga',
    'fuv': 'Fulfude Nigeria',
    'gaz': 'Oromo', #, West Central',
    'kea': 'Kabuverdianu',
    'nya': 'Nyanja',
    'ssw': 'Swati',
    'luo': 'Dholuo',# (Luo)',
    'ven': 'Venda',
    'kir':"Kirundi",
}

# ===== Build Language Name→ISOs map =====
def build_langname_to_isos(iso2name):
    name2isos = defaultdict(set)
    for iso, name in iso2name.items():
        name2isos[name].add(iso)
    return name2isos

def compare_models(model_1_name, model_2_name):
    """
    Prepares a DataFrame comparing the performance of two models task-by-task.
    """
    if model_1_name == model_2_name:
        return pd.DataFrame([{"Info": "Please select two different models to compare."}])

    # Get data for each model from the main leaderboard results
    df1 = all_df[(all_df['model'] == model_1_name) & (all_df['leaderboard'] == 'main')][['task', 'score', 'metric']].rename(columns={'score': model_1_name})
    df2 = all_df[(all_df['model'] == model_2_name) & (all_df['leaderboard'] == 'main')][['task', 'score']].rename(columns={'score': model_2_name})

    if df1.empty or df2.empty:
        return pd.DataFrame([{"Info": "One or both selected models have no 'main' leaderboard data to compare."}])

    # Merge the two dataframes on the task ID
    comp_df = pd.merge(df1, df2, on='task', how='outer')

    # Add descriptive columns
    comp_df['Cluster'] = comp_df['task'].map(TASK_TO_CLUSTER_MAP)
    comp_df['Task Name'] = comp_df['task'].map(TASKS_LIST)
    comp_df['Metric'] = comp_df['metric'].map(metrics_list)
    comp_df.fillna({'Cluster': 'Uncategorized'}, inplace=True)
    
    # Calculate the score difference, ensuring scores are numeric
    score1 = pd.to_numeric(comp_df[model_1_name], errors='coerce')
    score2 = pd.to_numeric(comp_df[model_2_name], errors='coerce')
    comp_df['Difference'] = score1 - score2

    # Format the difference column with colors
    def format_diff(d):
        if pd.isna(d):
            return "---"
        if d > 0.001:  # Model 1 is better
            return f"<span style='color:green; font-weight:bold;'>+{d:.2f}</span>"
        elif d < -0.001:  # Model 2 is better
            return f"<span style='color:red; font-weight:bold;'>{d:.2f}</span>"
        else:
            return f"{d:.2f}"

    # Format all score columns
    comp_df[model_1_name] = comp_df[model_1_name].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "---")
    comp_df[model_2_name] = comp_df[model_2_name].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "---")
    comp_df['Difference'] = comp_df['Difference'].apply(format_diff)

    # --- MODIFIED: Added 'task' to the list of final columns ---
    final_cols = ['Cluster', 'Task Name', 'task', 'Metric', model_1_name, model_2_name, 'Difference']
    comp_df = comp_df[final_cols]
    comp_df = comp_df.sort_values(by=['Cluster', 'Task Name']).reset_index(drop=True)

    # --- NEW: Renamed 'task' column to 'Task ID' for display ---
    comp_df.rename(columns={'task': 'Task ID'}, inplace=True)

    return comp_df
    
def get_model_table(model_name):
    """
    Generates a performance table for a specific model, showing cluster, task, and score.
    The table is sorted by Cluster and then by Task Name.
    """
    # Filter for the selected model and only 'main' leaderboard entries
    model_df = all_df[(all_df['model'] == model_name) & (all_df['leaderboard'] == 'main')].copy()

    if model_df.empty:
        return pd.DataFrame([{"Info": f"No 'main' leaderboard data available for the model: {model_name}"}])

    # --- NEW: Add the Cluster Name column using the map ---
    model_df['Cluster'] = model_df['task'].map(TASK_TO_CLUSTER_MAP)
    
    # Create other descriptive columns
    model_df['Task Name'] = model_df['task'].map(TASKS_LIST)
    model_df['Metric'] = model_df['metric'].map(metrics_list)
    model_df['Score'] = model_df['score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "---")

    # --- MODIFIED: Select the new 'Cluster' column for the final table ---
    table = model_df[['Cluster', 'Task Name', 'task', 'Metric', 'Score']].rename(columns={'task': 'Task ID'})

    # --- MODIFIED: Sort by Cluster first, then by Task Name ---
    table = table.sort_values(by=['Cluster', 'Task Name']).reset_index(drop=True)

    # Handle cases where a task might not be in a cluster
    table['Cluster'].fillna('Uncategorized', inplace=True)

    return table
    
def get_task_leaderboard(task_key):
    """
    Generates a leaderboard for a specific task, showing model performance across all languages.
    """
    # Filter the main DataFrame for the selected task
    task_df = all_df[all_df['task'] == task_key].copy()

    if task_df.empty:
        return pd.DataFrame([{"Info": f"No data available for the task: {TASKS_LIST.get(task_key, task_key)}"}])

    # Get the metric for this task to display later
    metric_name = metrics_list.get(task_df['metric'].iloc[0], '')

    # Create a user-friendly column name for each language/leaderboard
    def make_lang_col(row):
        lb = row['leaderboard']
        if lb == 'main':
            # Skip the 'main' leaderboard for task-specific views as it's an aggregate
            return None
        if '-' in lb:
            pair_lang = lb.split('-')
            # Handles cases where an ISO code might not be in our map
            src_lang = LANG_ISO2NAME.get(pair_lang[0], pair_lang[0])
            tgt_lang = LANG_ISO2NAME.get(pair_lang[1], pair_lang[1])
            return f"{src_lang} to {tgt_lang}"
        else:
            return LANG_ISO2NAME.get(lb, lb)
    if task_key not in ['lid']:
        task_df['lang_col'] = task_df.apply(make_lang_col, axis=1)
        task_df.dropna(subset=['lang_col'], inplace=True) # Remove rows where lang_col is None
    
        if task_df.empty:
            return pd.DataFrame([{"Info": f"No language-specific data for the task: {TASKS_LIST.get(task_key, task_key)}"}])

        # Pivot the table to have models as rows and languages as columns
        table = task_df.pivot_table(index='model', columns='lang_col', values='score', aggfunc='mean').reset_index()
    else:
        table = task_df.pivot_table(index='model', columns='task', values='score', aggfunc='mean').reset_index()
    
    score_cols = [col for col in table.columns if col != 'model']
    for col in score_cols:
        table[col] = table[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    main_score_map = all_df[(all_df['task'] == task_key) & (all_df['leaderboard'] == 'main')].set_index('model')['score']
    table.insert(1, 'Task Score', table['model'].map(main_score_map).apply(lambda x: f"{x:.2f}" if pd.notna(x) else "---"))

    # Add ranking medals based on the "Task Score"
    table = add_medals_to_models(table, score_col="Task Score")
    
    # Rename columns to be more descriptive, including the metric
    # rename_cols = {col: f"{col}<br>Metric: {metric_name}" for col in score_cols}
    if task_key in ['belebele', 'ner', 'mgsm', 'mmlu']:
        # rename_cols = {col: f"<div class='rotate_div'><br>{next(iter(LANGNAME2ISOS.get(col)))}</div>" for col in score_cols}
        rename_cols = {col: f"<div class='rotate_div'><br>{col}</div>" for col in score_cols}
    else:
        rename_cols = {col: f"{col}" for col in score_cols}
    table.rename(columns=rename_cols, inplace=True)

    return table
    
def get_task_metric_map(df):
    mapping = {}
    for _, row in df.iterrows():
        mapping[row["task"]] = row["metric"]
    return mapping

def cluster_average(row, tasks):
    vals = []
    for t in tasks:
        try:
            v = float(row[t])
            vals.append(v)
        except Exception:
            continue
    return np.mean(vals) if vals else np.nan

def add_medals_to_models(df, score_col="overall score"):
    score_float_col = "__score_float"
    df[score_float_col] = df[score_col].apply(lambda x: float(x) if x != "---" else np.nan)
    df = df.sort_values(by=score_float_col, ascending=False, kind="mergesort").reset_index(drop=True)
    def get_rank_symbols(scores):
        unique_scores = sorted(set([s for s in scores if not pd.isna(s)]), reverse=True)
        symbols = ["🏆", "🥈", "🥉"]
        score_to_symbol = {s: symbols[i] for i, s in enumerate(unique_scores[:3])}
        return [score_to_symbol.get(s, "") for s in scores]
    df['rank_symbol'] = get_rank_symbols(df[score_float_col].tolist())
    df['model'] = df['rank_symbol'] + ' ' + df['model']
    df = df.drop(columns=['rank_symbol', score_float_col])
    return df

def format_cluster_table(df, cluster_tasks, metric_map):
    col_order = ["model"] + cluster_tasks
    for t in cluster_tasks:
        if t not in df.columns:
            df[t] = '---'
    df = df[col_order]
    for t in cluster_tasks:
        df[t] = df[t].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.integer, np.floating)) else x)
    df["Cluster Score"] = df[cluster_tasks].apply(
        lambda row: cluster_average(row, cluster_tasks), axis=1
    )
    df["Cluster Score"] = df["Cluster Score"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "---")
    df = df[["model", "Cluster Score"] + cluster_tasks]
    # rename = {t: f"{t}\n{metric_map.get(t, '')}" for t in cluster_tasks}
    rename = {t: f"{TASKS_LIST[t]}<br>Metric: {metrics_list[metric_map.get(t, '')]}" for t in cluster_tasks}
    df = df.rename(columns=rename)
    df = add_medals_to_models(df, score_col="Cluster Score")
    return df

def format_main_overall_table(df, metric_map):
    main = df.copy()
    for cname, tasks in CLUSTERS.items():
        main[cname] = main[tasks].apply(lambda row: cluster_average(row, tasks), axis=1)
    cluster_cols = list(CLUSTERS.keys())
    main["Overall Score"] = main[cluster_cols].apply(
        lambda row: np.nanmean([x for x in row if pd.notna(x)]), axis=1
    )
    for c in cluster_cols + ["Overall Score"]:
        main[c] = main[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "---")
    main = main[["model", "Overall Score"] + cluster_cols]
    main = add_medals_to_models(main, score_col="Overall Score")
    main.rename(columns={'Overall Score': 'Sahara Score'}, inplace=True)
    return main

def load_leaderboards():
    df = load_private_leaderboard_df()
    metric_map = get_task_metric_map(df)
    main_df = df[df['leaderboard'] == 'main'].copy()
    if main_df.empty:
        cluster_tabs = {c: pd.DataFrame([{"Info": "No data"}]) for c in CLUSTERS}
        main_overall_tab = pd.DataFrame([{"Info": "No data"}])
        return cluster_tabs, main_overall_tab, [], {}, df, metric_map
    main_tasks_df = main_df.pivot_table(index='model', columns='task', values='score').reset_index()
    cluster_tabs = {}
    for cname, tasks in CLUSTERS.items():
        cluster_tabs[cname] = format_cluster_table(main_tasks_df, tasks, metric_map)
    for t in ALL_TASKS:
        if t not in main_tasks_df.columns:
            main_tasks_df[t] = np.nan
    main_overall_tab = format_main_overall_table(main_tasks_df, metric_map)
    all_langs = sorted([lb for lb in df['leaderboard'].unique() if lb not in ['main']])
    return cluster_tabs, main_overall_tab, df, metric_map

def df_to_html(df, col_minwidth=90, col_maxwidth=140, model_col_width=400):
    # Remove any column whose name contains "task"
    drop_cols = [col for col in df.columns if "task" in col]
    df = df.drop(columns=drop_cols, errors="ignore")
    df.columns.name = None   
    html = df.to_html(index=False, escape=False)
    return html



cluster_tabs, main_overall_tab, all_df, metric_map = load_leaderboards()

LANGNAME2ISOS = build_langname_to_isos(LANG_ISO2NAME)
#show only African langs
LANG_NAME_LIST = sorted([lang for lang in LANGNAME2ISOS.keys() if lang not in ['eng', 'fra', 'English', 'French']])
# TASK_NAME_LIST = sorted(list(TASKS_LIST.values()))
# Create a list of choices in the format "Task Name (id)"
TASK_NAME_LIST = sorted([f"{name} ({key})" for key, name in TASKS_LIST.items()])
TASK_NAME2KEY = {v: k for k, v in TASKS_LIST.items()}

# Get the list of unique model names for the new dropdown
MODEL_NAME_LIST = sorted(all_df['model'].unique()) if not all_df.empty else []

def get_lang_table(lang_name):
    iso_codes = LANGNAME2ISOS.get(lang_name, [])
    if not iso_codes:
        return pd.DataFrame([{"Info": "No data for this language"}])
    # Find all leaderboards containing any ISO in this language group
    pattern = re.compile(r"(^|-)(" + "|".join(re.escape(iso) for iso in iso_codes) + r")(-|$)")
    matched_langs = [lb for lb in all_df['leaderboard'].unique() if lb not in ['main'] and pattern.search(lb)]
    lang_df = all_df[all_df['leaderboard'].isin(matched_langs)].copy()
    if lang_df.empty:
        return pd.DataFrame([{"Info": "No data for this language"}])
    def make_task_col(row):
        lb = row['leaderboard']
        task = row['task']
        metric = row['metric']
        if '-' in lb:
            pair_lang = lb.split('-')
            pair = lb.replace('-', '_')
            # return f"{TASKS_LIST[task]}({task}) {LANG_ISO2NAME[pair_lang[0]]} to {LANG_ISO2NAME[pair_lang[1]]} ({pair})\n{metric}"
            return f"{TASKS_LIST[task]} <br> {LANG_ISO2NAME[pair_lang[0]]} to {LANG_ISO2NAME[pair_lang[1]]} <br> Metric: {metrics_list[metric]}"
        else:
            return f"{TASKS_LIST[task]} <br>  Metric: {metrics_list[metric]}"
    lang_df['task_col'] = lang_df.apply(make_task_col, axis=1)
    table = lang_df.pivot_table(index='model', columns='task_col', values='score').reset_index()
    score_cols = [col for col in table.columns if col != 'model']
    for col in score_cols:
        table[col] = table[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.integer, np.floating)) else x)
    def avg_score(row):
        vals = []
        for col in score_cols:
            try:
                v = float(row[col])
                vals.append(v)
            except Exception:
                continue
        return np.mean(vals) if vals else np.nan
    table.insert(1, 'Language Score', table.apply(avg_score, axis=1).apply(lambda x: f"{x:.2f}" if pd.notna(x) else "---"))
    table['__overall_score_float'] = table['Language Score'].apply(lambda x: float(x) if x != "---" else np.nan)
    table = table.sort_values(by='__overall_score_float', ascending=False, kind="mergesort").reset_index(drop=True)
    def get_rank_symbols(scores):
        unique_scores = sorted(set([s for s in scores if not pd.isna(s)]), reverse=True)
        symbols = ["🏆", "🥈", "🥉"]
        score_to_symbol = {s: symbols[i] for i, s in enumerate(unique_scores[:3])}
        return [score_to_symbol.get(s, "") for s in scores]
    table['rank_symbol'] = get_rank_symbols(table['__overall_score_float'].tolist())
    table['model'] = table['rank_symbol'] + ' ' + table['model']
    table = table.drop(columns=['rank_symbol', '__overall_score_float'])
    return table

