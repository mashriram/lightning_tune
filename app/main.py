import gradio as gr
import lightning_tune as lt
from pathlib import Path
import yaml, warnings


def analyze_dataset(model_id, dataset_file):
    if not model_id or dataset_file is None:
        raise gr.Error("Please specify a Model ID and upload a dataset first.")
    try:
        # Suppress the warning about the output column for the UI
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = lt.PipelineConfig.from_dataset(
                model_repo_id=model_id,
                file_path=Path(dataset_file.name),
                image_root_path=Path(dataset_file.name).parent,
            )

        vis_col = (
            config.data.vision_config.image_column if config.data.vision_config else ""
        )
        num_cols = (
            ", ".join(config.data.tabular_config.numerical_columns)
            if config.data.tabular_config
            else ""
        )
        cat_cols = (
            ", ".join(config.data.tabular_config.categorical_columns)
            if config.data.tabular_config
            else ""
        )
        txt_cols, tgt_col = (
            ", ".join(config.data.text_columns),
            config.data.output_column,
        )

        # Update UI components and make the next tabs visible
        return {
            config_state: config,
            # Make configuration tab selectable
            config_tab: gr.update(interactive=True),
            # Update values in configuration tab
            vis_col_out: vis_col,
            num_cols_out: num_cols,
            cat_cols_out: cat_cols,
            txt_cols_out: txt_cols,
            tgt_col_out: tgt_col,
            peft_method_in: config.train.peft.method,
            lora_r_in: config.train.peft.r,
            lora_alpha_in: config.train.peft.lora_alpha,
            lr_in: config.train.llm_lr,
            epochs_in: config.trainer.max_epochs,
        }
    except Exception as e:
        raise gr.Error(f"Failed to analyze dataset: {e}")


def start_finetuning_ui(
    config, v_col, n_cols, c_cols, t_cols, target_col, peft, r, alpha, lr, epochs
):
    if not isinstance(config, lt.PipelineConfig):
        raise gr.Error("Analyze a dataset first by going back to the 'Setup' tab.")

    # Update config with user's choices from the UI
    if config.data.vision_config:
        config.data.vision_config.image_column = v_col
    if config.data.tabular_config:
        config.data.tabular_config.numerical_columns = [
            c.strip() for c in n_cols.split(",") if c.strip()
        ]
        config.data.tabular_config.categorical_columns = [
            c.strip() for c in c_cols.split(",") if c.strip()
        ]
    config.data.text_columns = [c.strip() for c in t_cols.split(",") if c.strip()]
    config.data.output_column = target_col
    config.train.peft.method = peft
    config.train.peft.r = r
    config.train.peft.lora_alpha = alpha
    config.train.llm_lr = lr
    config.trainer.max_epochs = epochs

    # Save the final config and start the run
    output_dir = config.get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "final_ui_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f)

    yield f"✅ Config saved to {output_dir}. Starting finetuning...\n(Check terminal for real-time progress)"
    try:
        lt.run_finetuning(config)
        yield f"✅ Finetuning job finished! Model saved in:\n{output_dir}"
    except Exception as e:
        yield f"❌ ERROR during finetuning: {e}"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Lightning Tune: The Smart Finetuning UI")
    config_state = gr.State()

    with gr.Tabs() as tabs:
        with gr.TabItem("1. Setup", id=0) as setup_tab:
            model_id_in = gr.Textbox(
                label="Hugging Face Model ID",
                placeholder="e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            )
            dataset_in = gr.File(label="Upload Dataset (CSV, JSON, or JSONL)")
            analyze_button = gr.Button("Analyze Dataset", variant="primary")

        with gr.TabItem("2. Configuration", id=1, interactive=False) as config_tab:
            gr.Markdown("🔬 **Smart Analysis Results** (Edit if needed)")
            with gr.Accordion("Data Configuration", open=True):
                vis_col_out = gr.Textbox(label="Image Column")
                num_cols_out = gr.Textbox(label="Numerical Columns (comma-separated)")
                cat_cols_out = gr.Textbox(label="Categorical Columns (comma-separated)")
                txt_cols_out = gr.Textbox(
                    label="Text Feature Columns (comma-separated)"
                )
                tgt_col_out = gr.Textbox(label="Target/Output Column")
            with gr.Accordion("Training Configuration", open=True):
                peft_method_in = gr.Dropdown(
                    label="PEFT Method",
                    choices=["qlora", "lora", "dora"],
                    value="qlora",
                )
                with gr.Row():
                    lora_r_in = gr.Slider(
                        label="LoRA Rank (r)", minimum=1, maximum=64, step=1, value=8
                    )
                    lora_alpha_in = gr.Slider(
                        label="LoRA Alpha", minimum=1, maximum=128, step=1, value=32
                    )
                lr_in = gr.Slider(
                    label="Learning Rate",
                    minimum=1e-6,
                    maximum=1e-3,
                    value=5e-5,
                    step=1e-6,
                )
                epochs_in = gr.Slider(
                    label="Epochs", minimum=1, maximum=50, value=3, step=1
                )
            launch_button = gr.Button("Start Finetuning", variant="primary")

        with gr.TabItem("3. Run", id=2, interactive=False) as run_tab:
            log_output = gr.Textbox(
                label="Status Log", lines=20, interactive=False, autoscroll=True
            )

    # --- Event Listeners ---
    def on_analyze_success(config, *args):
        # This function receives the outputs of analyze_dataset
        # and returns the new selected tab index
        return gr.update(selected=1)

    analysis_outputs = {
        config_state,
        config_tab,
        vis_col_out,
        num_cols_out,
        cat_cols_out,
        txt_cols_out,
        tgt_col_out,
        peft_method_in,
        lora_r_in,
        lora_alpha_in,
        lr_in,
        epochs_in,
    }
    analyze_button.click(
        fn=analyze_dataset,
        inputs=[model_id_in, dataset_in],
        outputs=list(analysis_outputs),
    ).then(on_analyze_success, outputs=tabs)

    def on_launch_click():
        return gr.update(selected=2, interactive=True), ""

    launch_button.click(
        on_launch_click, outputs=[tabs, log_output]
    ).then(
        fn=start_finetuning_ui,
        inputs=[
            config_state,
            vis_col_out,
            num_cols_out,
            cat_cols_out,
            txt_cols_out,
            tgt_col_out,
            peft_method_in,
            lora_r_in,
            lora_alpha_in,
            lr_in,
            epochs_in,
        ],
        outputs=log_output,
    )

if __name__ == "__main__":
    demo.launch(share=True)
