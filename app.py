import gradio as gr

from src.helper import *


# Custom CSS to replicate the Google-style card design from the image
custom_head_html = """
<link rel="stylesheet" href="https://africa.dlnlp.ai/sahara/font-awesome/css/font-awesome.min.css">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.0/jquery.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
<link rel="stylesheet" type="text/css" href="./public/css/style.min.css">
<script defer src="./public/js/script.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Rubik:wght@400;600&display=swap" rel="stylesheet">
"""

new_header_html = """
<center>
          <br><br><br>
          <img src="https://africa.dlnlp.ai/sahara/img/sahara_web_main.jpg" alt="Sahara logo" width="60%"> 
        </p>
</center>
<br style="height:1px;">
"""

google_style_css = """
/* Add this rule to your google_style_css string */

/* Citation Block */
        .citation-block {
            position: relative;
            background-color: #FDF6E3 !important; /* Light cream background */
            border-radius: 8px;
            padding: 25px;
        }
        .citation-block pre {
            background-color: transparent;
            border: none;
            padding: 0;
            /* font-size: 14px; */
            white-space: pre-wrap;
            word-break: break-all;
        }
        .citation-block .btn-copy {
            position: absolute;
            top: 15px;
            right: 15px;
            background-color: #D97706 !important;
            border-color: #c56a05 !important;
            color: white !important;
        }
        .citation-block .btn-copy:hover, .citation-block .btn-copy:focus {
            background-color: #c56a05 !important;
            color: white !important;
        }


.fillable.svelte-15jxnnn.svelte-15jxnnn:not(.fill_width) {
       /* min-width: 400px !important; */
        max-width: 1580px !important;
    }
  
.flat-navy-button {
    background-color: #117b75 !important; /* Navy Blue */
    color: #fff !important;
    font-weight: bold !important;
    border-radius: 5px !important; /* Slightly rounded corners */
    border: none !important;
    box-shadow: none !important;
}
.flat-navy-button:hover {
    background-color: #117b75 !important; /* Lighter navy for hover */
    color: #e8850e !important;
}
div[class*="gradio-container"] {
background:#FFFBF5 !important;
color:#000 !important;
}

div.svelte-1nguped {
    background: white !important;
}
/* Main Content Area */
        .content-section {
            padding: 60px 0;
        }
        .content-card {
            background-color: #fff !important;
            border-radius: 12px;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
            padding: 40px;
            margin-bottom: 40px;
        }
.btn-cite {
            color: #7d3561;
            font-size: 16px;
            margin: 0 3px; /* Add spacing between multiple icons */
        }
        .content-card h4 {
          font-family: "Rubik", sans-serif;
          color: #7d3561 !important;
        }
        .content-card h2 {
          font-family: "Rubik", sans-serif;
          font-size: 30px;
          font-weight: 600;
          line-height: 1.25;
          letter-spacing: -1px;
          color: #2f3b7d !important;
          text-transform:none;

            /* font-size: 30px;
            font-weight: bold;
            color: #D97706; /* Brand Orange */
            margin-top: 0;
            margin-bottom: 20px; */
        }
        .content-card h3 {
          font-size: 20px;
          color: #2f3b7d !important;
        }
        .content-card h3 .title {
          color: #7d3561 !important;
        }
        .content-card p {
            /* font-size: 18px; */
            /* line-height: 1.7; */
        }

div.svelte-wv8on1{
    # border: 2px solid #074e4a !important; 
    border-top: 0 !important;
     /* background-color: #fff2eb !important; */
     padding: 10px !important;
}
.padding.svelte-phx28p {
    padding:0 !important;
}

.tab-wrapper.svelte-1tcem6n.svelte-1tcem6n {
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    height: 0 !important;
    padding-bottom: 0 !important;
}


.selected.svelte-1tcem6n.svelte-1tcem6n {
    background-color: #7d3561 !important;
    color: #fff !important;
}
.tabs.svelte-1tcem6n.svelte-1tcem6n {
    /* border: 1px solid #dca02a !important; */
    border-top: 0 !important;
    /* background-color: #dca02a !important; */
}
button.svelte-1tcem6n.svelte-1tcem6n {
    color: #7d3561 !important;
    /* border: 1px solid #dca02a !important; */
    font-weight: bold;
    /* font-size: 16px; */
    padding: 8px 5px;
    background-color: #fff !important;
}
button.svelte-1tcem6n.svelte-1tcem6n:hover {
    background-color: #de8fc2 !important;
}
.tab-container.svelte-1tcem6n.svelte-1tcem6n:after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 2px;
    background-color: #7d3561 !important;
}

div[class*="gradio-container"]  .prose table,
div[class*="gradio-container"]  .prose tr,
div[class*="gradio-container"]  .prose td,
div[class*="gradio-container"]  .prose th {
    border: 0 !important;
    border-top: 2px solid #dca02a;
    border-bottom: 2px solid #dca02a;
}


div[class*="gradio-container"]  .prose table {
    color:#000 !important;
    border-top: 2px solid #dca02a !important;
    border-bottom: 2px solid #dca02a !important;
    margin-bottom:20px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
}
div[class*="gradio-container"]  .prose thead tr {
    border-bottom: 2px solid #dca02a !important;
}
div[class*="gradio-container"] .prose th .rotate_div {
transform: rotate(-45deg) !important; 
color: #7d3561 !important;
padding-top: 10px !important;
padding-bottom: 5px !important;
font-weight: None !important;
font-size: 12px !important;
/*height: 30px !important;  Give the cell enough height for the rotated text */
}
div[class*="gradio-container"]  .prose th {
    /* transform: rotate(-90deg) !important; */
    color: #7d3561 !important;
    font-weight: bold;
    /* font-size: 20px; */
    padding: 8px 5px;
    vertical-align: middle;
    border: 0 !important;
}
div[class*="gradio-container"]  .prose td {
    /* font-size: 18px; */
    padding: 8px 5px;
    border: 0 !important;
    vertical-align: middle;
    color:#000 !important;
}
div[class*="gradio-container"]  .prose th:first-child,
div[class*="gradio-container"]  .prose td:first-child {
    transform: rotate(0deg) !important;
    min-width: 400px !important;
    /* max-width: 400px !important; */
    width:400px !important;
    text-align: left !important;
}
div[class*="gradio-container"]  .prose th:not(:first-child),
div[class*="gradio-container"]  .prose td:not(:first-child) {
    /* min-width: 90px;
    max-width: 140px;
    width:auto !important; */
    text-align: center;
}
/* --- CUSTOM RULE FOR THE SECOND CHILD --- */
#model_specific_table .prose td:nth-child(2) {
    text-align: left;         /* Example: A custom alignment */
}
/* --- Styles for the Model Comparison Table --- */

/* Rule for the Second Column (Task Name) */
#models_comparasion_table .prose th:nth-child(2),
#models_comparasion_table .prose td:nth-child(2) {
    width: 200px !important;  /* Give it enough width */
    text-align: left !important;
    white-space: nowrap;          /* Prevent text from wrapping */
}

/* Rule for the First Column (Cluster) */
#models_comparasion_table .prose th:first-child,
#models_comparasion_table .prose td:first-child {
    width: 130px !important;
    text-align: left !important;
}

/* Rule for other columns (Task ID, Metric, Scores, etc.) */
#models_comparasion_table .prose th:not(:nth-child(1)):not(:nth-child(2)),
#models_comparasion_table .prose td:not(:nth-child(1)):not(:nth-child(2)) {
    width: 95px !important;  /* Set a consistent width for other columns */
    text-align: center;
}

div[class*="gradio-container"] .md :not(pre)>code
 {
    background: #fcbb99 !important;
    border: 1px solid #763412 !important;
}
div[class*="gradio-container"] .md *
{
    color: #000 !important;
}
"""

introduction_text = """

"""
favicon_head = '<link rel="icon" type="image/x-icon" href="/file=favicon.ico">'

# with gr.Blocks(title="Sahara Leaderboard", css=custom_css) as demo:
# with gr.Blocks(title="Sahara Leaderboard") as demo:
with gr.Blocks(theme=gr.themes.Default(), title="Sahara Benchmark Leaderboards", css=google_style_css, head=favicon_head) as demo:
    # Use elem_classes to apply our custom CSS to this group
    gr.HTML(new_header_html)
    # === UPDATED BUTTONS START ===
    with gr.Row():
        gr.Button("Official Website", link="https://africa.dlnlp.ai/sahara", elem_classes=['flat-navy-button'])
        gr.Button("ACL 2025 Paper", link="https://aclanthology.org/2025.acl-long.1572", elem_classes=['flat-navy-button'])
        gr.Button("Tasks", link="https://africa.dlnlp.ai/sahara/tasks", elem_classes=['flat-navy-button'])
        # gr.Button("Submission Instructions", link="https://africa.dlnlp.ai/sahara/instructions", elem_classes=['flat-navy-button'])
        # gr.Button("New Submission", link="https://africa.dlnlp.ai/sahara/submit", elem_classes=['flat-navy-button'])
        gr.Button("HF Dataset Repo", link="https://huggingface.co/datasets/UBC-NLP/sahara_benchmark", elem_classes=['flat-navy-button'])
        gr.Button("GitHub Repo", link="https://github.com/UBC-NLP/sahara", elem_classes=['flat-navy-button'])
        # These buttons will now show an alert message
        # submission_btn = gr.Button("Submission Instructions", elem_classes=['flat-navy-button'])
        # new_submission_btn = gr.Button("New Submission", elem_classes=['flat-navy-button'])
        # github_btn = gr.Button("GitHub Repo", elem_classes=['flat-navy-button'])
        # # Function to display the alert
        # def show_coming_soon_alert():
        #     gr.Info("Stay tuned! It will be ready soon.")
        
        # # Link the click event of each button to the alert function
        # submission_btn.click(fn=show_coming_soon_alert)
        # new_submission_btn.click(fn=show_coming_soon_alert)
        # github_btn.click(fn=show_coming_soon_alert)
    # === UPDATED BUTTONS END ===
    
    with gr.Group(elem_classes="content-card"):
        gr.Markdown("<br>")
        # Hidden component to trigger JavaScript on load
        # url_trigger = gr.Textbox(visible=False)
        
        # State to hold URL parameters
        # url_params_state = gr.State({})
        
        with gr.Tabs() as tabs:
            # Main leaderboard
            with gr.Tab("Main Leaderboard", id="main"):
                gr.HTML("<br><br><center><h2>Main Leaderboard</h2></center><br>")
                gr.HTML(df_to_html(main_overall_tab))
            # Task Clusters leaderboards
            with gr.Tab("Task-Cluster Leaderboard", id="clusters"):
                gr.HTML("<br><br><center><h2>Task-Cluster Leaderboard</h2></center><br>")
                CLUSTERS_NAME=[cname for cname, cdf in cluster_tabs.items()]
                cname = CLUSTERS_NAME[0]
                initial_cluster_title_html = f"<h3><span class='title'>Task-Cluster name:</span> {cname}</span></h3>"

                # 2. Create a variable for the title component so it can be updated.
                cluster_title_component = gr.HTML(initial_cluster_title_html)
                
                clusters_dropdown = gr.Dropdown(
                    choices=CLUSTERS_NAME, 
                    label="Select Task-CLuster", 
                    interactive=True, 
                    elem_id="cluster_dropdown",
                    value=CLUSTERS_NAME[0]  # Set default value
                )
                
                def get_claster_table(cluster_name):
                    for cname, cdf in cluster_tabs.items():
                        if cname== cluster_name:
                            return cdf
                    return None
                    
                cluster_table_component = gr.HTML(df_to_html(get_claster_table(CLUSTERS_NAME[0])) if CLUSTERS_NAME else "<b>No cluser found</b>")
                
                def update_cluster_table(cluster_name):
                    df = get_claster_table(cluster_name)
                    cluster_title_html = f"<h3><span class='title'>Task-Cluster name:</span> {cluster_name}</span></h3>"
                    return cluster_title_html, df_to_html(df) if df is not None else "<b>No data found</b>"
                
                clusters_dropdown.change(update_cluster_table, clusters_dropdown, outputs=[cluster_title_component, cluster_table_component])
                
            # Languages Leaderboards
            # Task-Specific Leaderboards
            with gr.Tab("Task-Specific Leaderboard", id="tasks"):
                # --- MODIFIED ---
                # 1. Define the initial title based on the first task in the list.
                gr.HTML("<br><br><center><h2>Task-Specific Leaderboard (per langauge)</h2></center><br>")
                initial_task_name = TASK_NAME_LIST[0]
                tname=initial_task_name.split(' (')[0]
                tid=initial_task_name.split(' (')[-1].split(')')[0]
                initial_title_html = f"<h3><span class='title'>Task name:</span> {tname}<br> <span class='title'>Task identifier:</span> {tid}</h3>"

                # 2. Create a variable for the title component so it can be updated.
                task_title_component = gr.HTML(initial_title_html)
                
                # Dropdown for selecting the task (remains the same)
                task_dropdown = gr.Dropdown(choices=TASK_NAME_LIST, label="Select Task", interactive=True, value=initial_task_name)
                
                # --- MODIFIED ---
                # 3. Modify the update function to return TWO values: the new title and the new table.
                def update_task_table(task_name_with_id):
                    # Create the new dynamic title HTML
                    tname=task_name_with_id.split(' (')[0]
                    tid=task_name_with_id.split(' (')[-1].split(')')[0]
                    new_title = f"<h3><span class='title'>Task name:</span> {tname}<br> <span class='title'>Task identifier:</span> {tid}</h3>"
                    # new_title = f"<br><br><center><h2>{task_name_with_id} Leaderboard</h2></center><br>"
                    
                    # Parse the task key to get the data
                    task_key = task_name_with_id.split('(')[-1].strip(')')
                    df = get_task_leaderboard(task_key)
                    
                    # Return both the new title and the HTML for the table
                    return new_title, df_to_html(df)

                # Initial table display (remains the same)
                initial_task_key = initial_task_name.split('(')[-1].strip(')')
                task_table_component = gr.HTML(df_to_html(get_task_leaderboard(initial_task_key)))

                # --- MODIFIED ---
                # 4. Update the .change() event to send outputs to BOTH the title and table components.
                task_dropdown.change(
                    fn=update_task_table, 
                    inputs=task_dropdown, 
                    outputs=[task_title_component, task_table_component]
                )
                
            with gr.Tab("Language-Specific Leaderboard", id="langs"):
                gr.HTML("<br><br><center><h2>Language-Specific Leaderboard (per task)</h2></center><br>")
                lang_name=LANG_NAME_LIST[0]
                initial_lang_title_html = f"<h3><span class='title'>Language name:</span> {lang_name} <br> <span class='title'>Language ISO-3:</span> {next(iter(LANGNAME2ISOS.get(lang_name)))} </h3>"

                # 2. Create a variable for the title component so it can be updated.
                lang_title_component = gr.HTML(initial_lang_title_html)
                lang_dropdown = gr.Dropdown(choices=LANG_NAME_LIST, label="Select Language", interactive=True)
                lang_table_component = gr.HTML(df_to_html(get_lang_table(LANG_NAME_LIST[0])) if LANG_NAME_LIST else "<b>No languages found</b>")
                def update_lang_table(lang_name):
                    df = get_lang_table(lang_name)
                    new_title = f"<h3><span class='title'>Language name:</span> {lang_name} <br> <span class='title'>Language ISO-3:</span> {next(iter(LANGNAME2ISOS.get(lang_name)))}</h3>"
                    return new_title, df_to_html(df)
                lang_dropdown.change(update_lang_table, lang_dropdown,  outputs=[lang_title_component, lang_table_component])
            # --- NEW TAB FOR MODEL-SPECIFIC LEADERBOARD ---
            with gr.Tab("Model-Specific Leaderboard", id="models", elem_id="model_specific_table"):
                gr.HTML("<br><br><center><h2>Model-Specific Leaderboard (per task)</h2></center><br>")
                
                initial_model_name = MODEL_NAME_LIST[0]
                initial_model_title_html = f"<h3><span class='title'>Model name:</span> {initial_model_name}</h3>"
        
                # Component to display the dynamic title
                model_title_component = gr.HTML(initial_model_title_html)
                
                # Dropdown for selecting the model
                model_dropdown = gr.Dropdown(
                    choices=MODEL_NAME_LIST, 
                    label="Select Model", 
                    interactive=True,
                    value=initial_model_name
                )
                
                # Component to display the model's performance table
                model_table_component = gr.HTML(df_to_html(get_model_table(initial_model_name)))
        
                # Function to update the title and table based on dropdown selection
                def update_model_table(model_name):
                    df = get_model_table(model_name)
                    new_title = f"<h3><span class='title'>Model name:</span> {model_name}</h3>"
                    return new_title, df_to_html(df)
        
                # Link the dropdown's change event to the update function
                model_dropdown.change(
                    fn=update_model_table, 
                    inputs=model_dropdown, 
                    outputs=[model_title_component, model_table_component]
                )
            # --- NEW TAB TO COMPARE MODELS ---
            with gr.Tab("Compare Models", id="compare"):
                gr.HTML("<br><br><center><h2>Compare Two Models</h2></center><br>")
                with gr.Row():
                    model_1_dd = gr.Dropdown(MODEL_NAME_LIST, label="Select Model 1", interactive=True)
                    model_2_dd = gr.Dropdown(MODEL_NAME_LIST, label="Select Model 2", interactive=True)
                compare_btn = gr.Button("Compare")

                # Note for the 'Difference' column
                explanation_note = """
                **Note on the 'Difference' Column:**
                * This value is calculated as: `(Score of Model 1) - (Score of Model 2)`.
                * A positive value in <span style='color:green; font-weight:bold;'>green</span> indicates that **Model 1** performed better on that task.
                * A negative value in <span style='color:red; font-weight:bold;'>red</span> indicates that **Model 2** performed better on that task.
                """
                # --- MODIFIED: Make the note invisible by default ---
                explanation_note_md = gr.Markdown(explanation_note, visible=False)

                # Create a container with a unique ID for the comparison table
                with gr.Column(elem_id="models_comparasion_table"):
                    comparison_output = gr.HTML("<p style='text-align:center;'>Select two models and click Compare to see the results.</p>")

                # --- MODIFIED: The function now returns TWO values (visibility and html) ---
                def update_comparison_table(m1, m2):
                    if not m1 or not m2:
                        gr.Info("Please select both models before clicking Compare.")
                        # Return an update to hide the note and the placeholder text
                        return gr.update(visible=False), "<p style='text-align:center;'>Please select two models to compare.</p>"
                    
                    df = compare_models(m1, m2)
                    # Return an update to SHOW the note and the results table
                    return gr.update(visible=True), df_to_html(df)

                # --- MODIFIED: Update the outputs list to target both components ---
                compare_btn.click(
                    fn=update_comparison_table,
                    inputs=[model_1_dd, model_2_dd],
                    outputs=[explanation_note_md, comparison_output]
                )
    with gr.Group(elem_classes="content-card"):
        gr.Markdown("<br>")
        gr.HTML("<h2>Citation</h2>If you use the Sahara benchmark for your scientific publication, or if you find the resources in this website useful, please cite our <a href='https://africa.dlnlp.ai/sahara/'>ACL2025 paper </a>.")
    gr.HTML("<center><img src='https://africa.dlnlp.ai/sahara//img/sahara_web_sponsers.jpg' width='25%'> </center>")

    
if __name__ == "__main__":
    demo.launch(share=True)