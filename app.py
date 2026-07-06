import os
os.environ["GRADIO_SSR_MODE"] = "False"

import gradio as gr
from orchestrator import run_orchestrator

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap');

body { font-family: 'Outfit', sans-serif !important; }

.terminal-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #0d1117 !important;
    color: #58a6ff !important;
    border: 1px solid #30363d !important;
    font-size: 0.85rem !important;
}

.title-header {
    text-align: center;
    padding: 1.5rem 1rem 0.25rem;
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 800;
}

.subtitle-header {
    text-align: center;
    color: #64748b;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

.space-link-box {
    padding: 0.75rem 1rem;
    background: linear-gradient(135deg, #1a1f35 0%, #0f172a 100%);
    border: 1px solid #3b82f6;
    border-radius: 8px;
    font-size: 1rem;
}

.deploy-section {
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 0.5rem;
    margin-top: 0.5rem;
}
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    block_border_color="*neutral_800",
    button_primary_background_fill="linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%)",
    button_primary_text_color="white",
)


def process_model(model_file, max_retries, hf_token, space_id):
    if model_file is None:
        yield "Error: Please upload a model file.", None, None, ""
        return

    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = model_file.name

    log_accumulator = "[Orchestrator] Starting ML Deployment Concierge...\n"
    log_accumulator += f"[Orchestrator] Model: {os.path.basename(model_path)} ({os.path.getsize(model_path)/1024:.1f} KB)\n"
    if hf_token and space_id:
        log_accumulator += f"[Orchestrator] HF Deploy target: {space_id.strip()}\n"
    log_accumulator += "-" * 70 + "\n"
    yield log_accumulator, None, None, ""

    agent_icons = {"Orchestrator": "[Orchestrator]", "Builder": "[Builder]", "Validator": "[Validator]"}

    try:
        for event in run_orchestrator(
            model_path,
            workspace_dir,
            max_retries=int(max_retries),
            hf_token=hf_token.strip() if hf_token else "",
            space_id=space_id.strip() if space_id else "",
        ):
            tag = agent_icons.get(event.get("agent", ""), "[Agent]")

            if event["type"] == "status":
                log_accumulator += f"{tag} {event['message']}\n"

            elif event["type"] == "error":
                log_accumulator += f"\n[FAILED] {event['message']}\n"
                log_file = event.get("log_file")
                yield log_accumulator, None, log_file, ""
                return

            elif event["type"] == "success":
                space_url = event.get("space_url", "")
                log_accumulator += f"\n[SUCCESS] {event['message']}\n"
                zip_path = event.get("zip_path")
                log_file = event.get("log_file")
                yield log_accumulator, zip_path, log_file, space_url
                return

            yield log_accumulator, None, None, ""

    except Exception as e:
        log_accumulator += f"\n[PANIC] Pipeline crashed: {str(e)}\n"
        yield log_accumulator, None, None, ""



def build_interface():
    with gr.Blocks(title="ML Deployment Concierge") as demo:
        gr.HTML("<div class='title-header'>ML Deployment Concierge</div>")
        gr.HTML("<div class='subtitle-header'>Upload a model → agents build, validate & deploy it to HuggingFace Spaces.</div>")

        with gr.Row(equal_height=False):
            # ── Left column: inputs ──────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 1. Model File")
                model_input = gr.File(
                    label="Upload model (.pkl, .keras, .h5, .pt, .pth)",
                    file_types=[".pkl", ".keras", ".h5", ".pt", ".pth", ".bin"],
                )

                max_retries = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="Max self-correction retries",
                )

                gr.Markdown("### 2. Deploy to HuggingFace Spaces *(optional)*")
                with gr.Group(elem_classes="deploy-section"):
                    hf_token = gr.Textbox(
                        label="HF Write Token",
                        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                        type="password",
                        info="Generate at huggingface.co/settings/tokens (write access)",
                    )
                    space_id = gr.Textbox(
                        label="Space ID",
                        placeholder="your-username/my-model-space",
                        info="Will be created if it doesn't exist. Leave blank to skip.",
                    )

                run_btn = gr.Button("Build & Validate Bundle", variant="primary", size="lg")

            # ── Right column: outputs ────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 3. Live Agent Log")
                log_output = gr.Textbox(
                    label="Agent Terminal",
                    lines=18,
                    max_lines=30,
                    placeholder="Upload a model and click Build to start...",
                    elem_classes="terminal-box",
                )

                space_link = gr.Textbox(
                    label="Live HuggingFace Space URL",
                    placeholder="(appears here after successful deploy)",
                    interactive=False,
                    elem_classes="space-link-box",
                )

                with gr.Row():
                    bundle_download = gr.File(
                        label="Download Deployment Bundle (ZIP)",
                        interactive=False,
                    )
                    log_download = gr.File(
                        label="Download Run Log (JSON)",
                        interactive=False,
                    )

        run_btn.click(
            fn=process_model,
            inputs=[model_input, max_retries, hf_token, space_id],
            outputs=[log_output, bundle_download, log_download, space_link],
        )

        # Quick-start examples
        gr.Markdown("### Try a pre-seeded eval case")
        gr.Examples(
            examples=[
                [os.path.join("eval_cases", "clean_sklearn_model.pkl"),  3, "", ""],
                [os.path.join("eval_cases", "dependency_conflict_model.keras"), 3, "", ""],
                [os.path.join("eval_cases", "corrupted_model.keras"),    3, "", ""],
                [os.path.join("eval_cases", "oversized_model.bin"),      3, "", ""],
            ],
            inputs=[model_input, max_retries, hf_token, space_id],
            label="Select a test case",
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=THEME, css=CSS)
