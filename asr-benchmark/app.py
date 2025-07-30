import gradio as gr
import pandas as pd
from datasets import load_dataset
from jiwer import wer, cer
import os
from datetime import datetime
import re

from huggingface_hub import login

# Login to Hugging Face Hub (if token is available)
token = os.environ.get("HG_TOKEN")
if token:
    login(token)


try:
    dataset = load_dataset("sudoping01/bambara-speech-recognition-benchmark", name="default")["eval"]
    references = {row["id"]: row["text"] for row in dataset}
    print(f"Loaded {len(references)} reference transcriptions")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    references = {}


leaderboard_file = "leaderboard.csv"
if not os.path.exists(leaderboard_file):

    sample_data = [
          ["test_1", 0.2264, 0.1094, 0.1922, "2025-03-15 10:30:45"],
         ["test_2", 0.3264, 0.1094, 0.1922, "2025-03-15 10:30:45"],
        ]
    pd.DataFrame(sample_data, 
                 columns=["Model_Name", "WER", "CER", "Combined_Score", "timestamp"]).to_csv(leaderboard_file, index=False)
    print(f"Created new leaderboard file with sample data")
else:
    leaderboard_df = pd.read_csv(leaderboard_file)
    

    if "Combined_Score" not in leaderboard_df.columns:
        leaderboard_df["Combined_Score"] = leaderboard_df["WER"] * 0.7 + leaderboard_df["CER"] * 0.3
        leaderboard_df.to_csv(leaderboard_file, index=False)
        print(f"Added Combined_Score column to existing leaderboard")
    print(f"Loaded leaderboard with {len(leaderboard_df)} entries")

def normalize_text(text):
    """Normalize text for WER/CER calculation"""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_metrics(predictions_df):
    """Calculate WER and CER for predictions."""
    results = []
    total_ref_words = 0
    total_ref_chars = 0

    for _, row in predictions_df.iterrows():
        id_val = row["id"]
        if id_val not in references:
            continue
            
        reference = normalize_text(references[id_val])
        hypothesis = normalize_text(row["text"])
        
        if not reference or not hypothesis:
            continue
            
        reference_words = reference.split()
        hypothesis_words = hypothesis.split()
        reference_chars = list(reference)
        
        try:
            sample_wer = wer(reference, hypothesis)
            sample_cer = cer(reference, hypothesis)
            
            sample_wer = min(sample_wer, 2.0)  
            sample_cer = min(sample_cer, 2.0)  
            
            total_ref_words += len(reference_words)
            total_ref_chars += len(reference_chars)
            
            results.append({
                "id": id_val,
                "reference": reference,
                "hypothesis": hypothesis,
                "ref_word_count": len(reference_words),
                "ref_char_count": len(reference_chars),
                "wer": sample_wer,
                "cer": sample_cer
            })
        except Exception as e:
            print(f"Error processing sample {id_val}: {str(e)}")
            pass
    
    if not results:
        raise ValueError("No valid samples for WER/CER calculation")
        
    avg_wer = sum(item["wer"] for item in results) / len(results)
    avg_cer = sum(item["cer"] for item in results) / len(results)
    

    weighted_wer = sum(item["wer"] * item["ref_word_count"] for item in results) / total_ref_words
    weighted_cer = sum(item["cer"] * item["ref_char_count"] for item in results) / total_ref_chars
    
    return avg_wer, avg_cer, weighted_wer, weighted_cer, results

def format_as_percentage(value):
    """Convert decimal to percentage with 2 decimal places"""
    return f"{value * 100:.2f}%"

def prepare_leaderboard_for_display(df, sort_by="Combined_Score"):
    """Format leaderboard for display with ranking and percentages"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Rank", "Model_Name", "WER (%)", "CER (%)", "Combined_Score (%)", "timestamp"])
    

    display_df = df.copy()
    

    display_df = display_df.sort_values(sort_by)
    
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))
    
    for col in ["WER", "CER", "Combined_Score"]:
        if col in display_df.columns:
            display_df[f"{col} (%)"] = display_df[col].apply(lambda x: f"{x * 100:.2f}")
    

    
    return display_df

def update_ranking(method):
    """Update leaderboard ranking based on selected method"""
    try:
        current_lb = pd.read_csv(leaderboard_file)
        
        if "Combined_Score" not in current_lb.columns:
            current_lb["Combined_Score"] = current_lb["WER"] * 0.7 + current_lb["CER"] * 0.3
        
        sort_column = "Combined_Score"
        if method == "WER Only":
            sort_column = "WER"
        elif method == "CER Only":
            sort_column = "CER"
        
        return prepare_leaderboard_for_display(current_lb, sort_column)
        
    except Exception as e:
        print(f"Error updating ranking: {str(e)}")
        return pd.DataFrame(columns=["Rank", "Model_Name", "WER (%)", "CER (%)", "Combined_Score (%)", "timestamp"])

def process_submission(model_name, csv_file):
    """Process a new model submission"""
    if not model_name or not model_name.strip():
        return "Error: Please provide a model name.", None
        
    if not csv_file:
        return "Error: Please upload a CSV file.", None
    
    try:
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            return "Error: Uploaded CSV is empty.", None
            
        if set(df.columns) != {"id", "text"}:
            return f"Error: CSV must contain exactly 'id' and 'text' columns. Found: {', '.join(df.columns)}", None
            
        if df["id"].duplicated().any():
            dup_ids = df[df["id"].duplicated()]["id"].unique()
            return f"Error: Duplicate IDs found: {', '.join(map(str, dup_ids[:5]))}", None

        missing_ids = set(references.keys()) - set(df["id"])
        extra_ids = set(df["id"]) - set(references.keys())
        
        if missing_ids:
            return f"Error: Missing {len(missing_ids)} IDs in submission. First few missing: {', '.join(map(str, list(missing_ids)[:5]))}", None
            
        if extra_ids:
            return f"Error: Found {len(extra_ids)} extra IDs not in reference dataset. First few extra: {', '.join(map(str, list(extra_ids)[:5]))}", None
        
        try:
            avg_wer, avg_cer, weighted_wer, weighted_cer, detailed_results = calculate_metrics(df)
            
            # Check for suspiciously low values
            if avg_wer < 0.001:
                return "Error: WER calculation yielded suspicious results (near-zero). Please check your submission CSV.", None
                
        except Exception as e:
            return f"Error calculating metrics: {str(e)}", None
        

        leaderboard = pd.read_csv(leaderboard_file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        combined_score = avg_wer * 0.7 + avg_cer * 0.3
        
        if model_name in leaderboard["Model_Name"].values:
            idx = leaderboard[leaderboard["Model_Name"] == model_name].index
            leaderboard.loc[idx, "WER"] = avg_wer
            leaderboard.loc[idx, "CER"] = avg_cer
            leaderboard.loc[idx, "Combined_Score"] = combined_score
            leaderboard.loc[idx, "timestamp"] = timestamp
            updated_leaderboard = leaderboard
        else:
            new_entry = pd.DataFrame(
                [[model_name, avg_wer, avg_cer, combined_score, timestamp]],
                columns=["Model_Name", "WER", "CER", "Combined_Score", "timestamp"]
            )
            updated_leaderboard = pd.concat([leaderboard, new_entry])
        
        updated_leaderboard = updated_leaderboard.sort_values("Combined_Score")
        updated_leaderboard.to_csv(leaderboard_file, index=False)
        
        display_leaderboard = prepare_leaderboard_for_display(updated_leaderboard)
        
        return f"Submission processed successfully! WER: {format_as_percentage(avg_wer)}, CER: {format_as_percentage(avg_cer)}, Combined Score: {format_as_percentage(combined_score)}", display_leaderboard
        
    except Exception as e:
        return f"Error processing submission: {str(e)}", None

def get_current_leaderboard():
    """Get the current leaderboard data for display"""
    try:
        if os.path.exists(leaderboard_file):
            current_leaderboard = pd.read_csv(leaderboard_file)
            
            if "Combined_Score" not in current_leaderboard.columns:
                current_leaderboard["Combined_Score"] = current_leaderboard["WER"] * 0.7 + current_leaderboard["CER"] * 0.3
                current_leaderboard.to_csv(leaderboard_file, index=False)
                
            return current_leaderboard
        else:
            return pd.DataFrame(columns=["Model_Name", "WER", "CER", "Combined_Score", "timestamp"])
    except Exception as e:
        print(f"Error getting leaderboard: {str(e)}")
        return pd.DataFrame(columns=["Model_Name", "WER", "CER", "Combined_Score", "timestamp"])

def create_leaderboard_table():
    """Create and format the leaderboard table for display"""
    leaderboard_data = get_current_leaderboard()
    return prepare_leaderboard_for_display(leaderboard_data)

with gr.Blocks(title="Bambara ASR Leaderboard") as demo:
    gr.Markdown(
        """
        # 🇲🇱 Bambara ASR Leaderboard
        
        This leaderboard tracks and evaluates speech recognition models for the Bambara language.
        Models are ranked based on Word Error Rate (WER), Character Error Rate (CER), and a combined score.
        
        ## Current Models Performance
        """
    )
    
    current_data = get_current_leaderboard()
    

    if len(current_data) > 0:
        best_model = current_data.sort_values("Combined_Score").iloc[0]
        gr.Markdown(f"""
        ### 🏆 Current Best Model: **{best_model['Model_Name']}**
        * WER: **{best_model['WER']*100:.2f}%**
        * CER: **{best_model['CER']*100:.2f}%**
        * Combined Score: **{best_model['Combined_Score']*100:.2f}%**
        """)
    
    with gr.Tabs() as tabs:
        with gr.TabItem("🏅 Model Rankings"):

            initial_leaderboard = create_leaderboard_table()
            
            ranking_method = gr.Radio(
                ["Combined Score (WER 70%, CER 30%)", "WER Only", "CER Only"], 
                label="Ranking Method",
                value="Combined Score (WER 70%, CER 30%)"
            )
            
            leaderboard_view = gr.DataFrame(
                value=initial_leaderboard,
                interactive=False,
                label="Models are ranked by selected metric - lower is better"
            )
            
            ranking_method.change(
                fn=update_ranking,
                inputs=[ranking_method],
                outputs=[leaderboard_view]
            )
            
            with gr.Accordion("Metrics Explanation", open=False):
                gr.Markdown(
                    """
                    ## Understanding ASR Metrics
                    
                    ### Word Error Rate (WER)
                    WER measures how accurately the ASR system recognizes whole words:
                    * Lower values indicate better performance
                    * Calculated as: (Substitutions + Insertions + Deletions) / Total Words
                    * A WER of 0% means perfect transcription
                    * A WER of 20% means approximately 1 in 5 words contains an error
                    
                    ### Character Error Rate (CER)
                    CER measures accuracy at the character level:
                    * More fine-grained than WER
                    * Better at capturing partial word matches
                    * Particularly useful for agglutinative languages like Bambara
                    
                    ### Combined Score
                    * Weighted average: 70% WER + 30% CER
                    * Provides a balanced evaluation of model performance
                    * Used as the primary ranking metric
                    """
                )
        
        with gr.TabItem("📊 Submit New Results"):
            gr.Markdown(
                """
                ### Submit a new model for evaluation
                
                Upload a CSV file with the following format:
                * Must contain exactly two columns: 'id' and 'text'
                * The 'id' column should match the reference dataset IDs
                * The 'text' column should contain your model's transcriptions
                """
            )
            
            with gr.Row():
                model_name_input = gr.Textbox(
                    label="Model Name", 
                    placeholder="e.g., MALIBA-AI/bambara-asr"
                )
                gr.Markdown("*Use a descriptive name to identify your model*")
            
            with gr.Row():
                csv_upload = gr.File(
                    label="Upload CSV File", 
                    file_types=[".csv"]
                )
                gr.Markdown("*CSV with columns: id, text*")
                
            submit_btn = gr.Button("Submit", variant="primary")
            output_msg = gr.Textbox(label="Status", interactive=False)
            leaderboard_display = gr.DataFrame(
                label="Updated Leaderboard",
                value=initial_leaderboard,
                interactive=False
            )
            
            submit_btn.click(
                fn=process_submission,
                inputs=[model_name_input, csv_upload],
                outputs=[output_msg, leaderboard_display]
            )
            
        with gr.TabItem("📝 Benchmark Dataset"):
            gr.Markdown(
                """
                ## About the Benchmark Dataset
                
                This leaderboard uses the **[sudoping01/bambara-speech-recognition-benchmark](https://huggingface.co/datasets/MALIBA-AI/bambara-speech-recognition-leaderboard)** dataset:
                
                * Contains diverse Bambara speech samples
                * Includes various speakers, accents, and dialects
                * Covers different speech styles and recording conditions
                * Transcribed and validated
                
                ### How to Generate Predictions
                
                To submit results to this leaderboard:
                
                1. Download the audio files from the benchmark dataset
                2. Run your ASR model on the audio files
                3. Generate a CSV file with 'id' and 'text' columns
                4. Submit your results using the form in the "Submit New Results" tab
                
                ### Evaluation Guidelines
                
                * Text is normalized (lowercase, punctuation removed) before metrics calculation
                * Extreme outliers are capped to prevent skewing results
                * All submissions are validated for format and completeness

                NB: This work is a collaboration between MALIBA-AI, RobotsMali AI4D-LAB and Djelia
                """
            )
            
    gr.Markdown(
        """
        ---
        ### About MALIBA-AI
        
        **MALIBA-AI: Empowering Mali's Future Through Community-Driven AI Innovation**
        
        *"No Malian Language Left Behind"*
        
        This leaderboard is maintained by the MALIBA-AI initiative to track progress in Bambara speech recognition technology.
        For more information, visit [MALIBA-AI on Hugging Face](https://huggingface.co/MALIBA-AI).
        """
    )

if __name__ == "__main__":
    demo.launch()