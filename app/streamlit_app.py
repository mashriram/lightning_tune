import streamlit as st
import lightning_tune as lt
from pathlib import Path
import yaml
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="⚡ Lightning Tune", layout="wide")

st.title("⚡ Lightning Tune: The Smart Finetuning UI")

if "config" not in st.session_state:
    st.session_state.config = None

# --- Sidebar for Setup ---
with st.sidebar:
    st.header("1. Setup")
    model_id = st.text_input("Hugging Face Model ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    dataset_file = st.file_uploader("Upload Dataset (CSV, JSON, or JSONL)")

    if st.button("Analyze Dataset"):
        if not model_id or not dataset_file:
            st.error("Please specify a Model ID and upload a dataset first.")
        else:
            try:
                # Save the uploaded file to a temporary location
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                file_path = temp_dir / dataset_file.name
                with open(file_path, "wb") as f:
                    f.write(dataset_file.getbuffer())

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    config = lt.PipelineConfig.from_dataset(
                        model_repo_id=model_id,
                        file_path=file_path,
                        image_root_path=temp_dir,
                    )
                st.session_state.config = config
                st.success("Analysis complete! Configure the run below.")
            except Exception as e:
                st.error(f"Failed to analyze dataset: {e}")

# --- Main Area for Configuration and Run ---
if st.session_state.config:
    config = st.session_state.config
    st.header("2. Configuration")

    with st.expander("Data Configuration", expanded=True):
        if config.data.vision_config:
            config.data.vision_config.image_column = st.text_input(
                "Image Column", config.data.vision_config.image_column
            )
        if config.data.tabular_config:
            config.data.tabular_config.numerical_columns = st.text_input(
                "Numerical Columns", ", ".join(config.data.tabular_config.numerical_columns)
            ).split(",")
            config.data.tabular_config.categorical_columns = st.text_input(
                "Categorical Columns", ", ".join(config.data.tabular_config.categorical_columns)
            ).split(",")
        config.data.text_columns = st.text_input(
            "Text Feature Columns", ", ".join(config.data.text_columns)
        ).split(",")
        config.data.output_column = st.text_input("Target/Output Column", config.data.output_column)

    with st.expander("Training Configuration", expanded=True):
        config.train.peft.method = st.selectbox(
            "PEFT Method", ["qlora", "lora", "dora"], index=["qlora", "lora", "dora"].index(config.train.peft.method)
        )
        config.train.peft.r = st.slider("LoRA Rank (r)", 1, 64, config.train.peft.r)
        config.train.peft.lora_alpha = st.slider("LoRA Alpha", 1, 128, config.train.peft.lora_alpha)
        config.train.llm_lr = st.slider(
            "Learning Rate", 1e-6, 1e-3, config.train.llm_lr, format="%.1e"
        )
        config.trainer.max_epochs = st.slider("Epochs", 1, 50, config.trainer.max_epochs)

    st.header("3. Run")
    if st.button("Start Finetuning"):
        try:
            output_dir = config.get_output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "final_streamlit_config.yaml", "w") as f:
                yaml.dump(config.model_dump(mode="json"), f)
            st.info(f"Config saved to {output_dir}. Starting finetuning...")

            # Placeholder for logging from the finetuning process
            log_placeholder = st.empty()
            log_placeholder.text("Finetuning in progress... (Check terminal for real-time logs)")

            lt.run_finetuning(config)

            st.success(f"Finetuning job finished! Model saved in:\n{output_dir}")
        except Exception as e:
            st.error(f"ERROR during finetuning: {e}")
else:
    st.info("Upload a dataset and click 'Analyze Dataset' to get started.")
