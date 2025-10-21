import gradio as gr
import lightning_tune as lt
from pathlib import Path
import yaml


def analyze_dataset(model_id, dataset_file):
    if not model_id or dataset_file is None:
        raise gr.Error("Please specify a Model ID and upload a dataset first.")
    try:
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
        return {
            config_state: config,
            analysis_group: gr.update(visible=True),
            vis_col_out: vis_col,
            num_cols_out: num_cols,
            cat_cols_out: cat_cols,
            txt_cols_out: txt_cols,
            tgt_col_out: tgt_col,
            lr_out: f"{config.train.llm_lr:.1e}",
            epochs_out: str(config.trainer.max_epochs),
            rank_out: str(config.train.peft.r),
        }
    except Exception as e:
        raise gr.Error(f"Failed to analyze dataset: {e}")


def start_finetuning_ui(config, v_col, n_cols, c_cols, t_cols, target_col, epochs):
    if not isinstance(config, lt.PipelineConfig):
        raise gr.Error("Analyze a dataset first.")
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
    config.trainer.max_epochs = epochs
    output_dir = config.get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "final_ui_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f)
    yield f"✅ Config saved. Starting finetuning...\n(Check terminal for progress)"
    try:
        lt.run_finetuning(config)
        yield f"✅ Finetuning job finished! Model saved in:\n{output_dir}"
    except Exception as e:
        yield f"❌ ERROR during finetuning: {e}"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Lightning Tune: The Smart Finetuning UI")
    config_state = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Setup")
            model_id_in = gr.Textbox(
                label="Hugging Face Model ID",
                placeholder="e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            )
            dataset_in = gr.File(label="Upload Dataset (CSV)")
            analyze_button = gr.Button("Analyze Dataset", variant="secondary")
        with gr.Column(scale=2):
            gr.Markdown("## 2. Configuration & Launch")
            with gr.Group(visible=False) as analysis_group:
                gr.Markdown("🔬 **Analysis Results** (Edit if needed)")
                vis_col_out = gr.Textbox(label="Image Column")
                num_cols_out = gr.Textbox(label="Numerical Columns (comma-separated)")
                cat_cols_out = gr.Textbox(label="Categorical Columns (comma-separated)")
                txt_cols_out = gr.Textbox(
                    label="Text Feature Columns (comma-separated)"
                )
                tgt_col_out = gr.Textbox(label="Target/Output Column")
                gr.Markdown("🚀 **Smart Hyperparameters**")
                with gr.Row():
                    lr_out = gr.Textbox(label="Learning Rate", interactive=False)
                    epochs_out = gr.Textbox(label="Epochs", interactive=False)
                    rank_out = gr.Textbox(label="LoRA Rank", interactive=False)
                epochs_in = gr.Slider(
                    label="Override Epochs", minimum=1, maximum=20, value=3, step=1
                )
                launch_button = gr.Button("Start Finetuning", variant="primary")
    gr.Markdown("## 3. Log")
    log_output = gr.Textbox(label="Status", lines=10, interactive=False)
    analyze_button.click(
        fn=analyze_dataset,
        inputs=[model_id_in, dataset_in],
        outputs=[
            config_state,
            analysis_group,
            vis_col_out,
            num_cols_out,
            cat_cols_out,
            txt_cols_out,
            tgt_col_out,
            lr_out,
            epochs_out,
            rank_out,
        ],
    )
    launch_button.click(
        fn=start_finetuning_ui,
        inputs=[
            config_state,
            vis_col_out,
            num_cols_out,
            cat_cols_out,
            txt_cols_out,
            tgt_col_out,
            epochs_in,
        ],
        outputs=[log_output],
    )
if __name__ == "__main__":
    demo.launch(share=True)
