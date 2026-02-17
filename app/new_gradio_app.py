import gradio as gr
import requests
import json
import websockets
import asyncio
import threading
import base64
from typing import Optional

import os

API_URL = os.getenv("LIGHTNING_TUNE_API_URL", "http://127.0.0.1:8000")
WS_URL = API_URL.replace("http", "ws")

# State
job_id_state = gr.State()
config_state = gr.State()
service_url_state = gr.State()

def search_models_api(query, token):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(f"{API_URL}/models", params={"query": query, "limit": 10}, headers=headers)
        if resp.status_code == 200:
            return [m["id"] for m in resp.json()]
        gr.Warning(f"Failed to search models: {resp.text}")
        return []
    except Exception as e:
        gr.Warning(f"Connection error: {e}")
        return []

def search_datasets_api(query, token):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(f"{API_URL}/datasets", params={"query": query, "limit": 10}, headers=headers)
        if resp.status_code == 200:
            return [d["id"] for d in resp.json()]
        gr.Warning(f"Failed to search datasets: {resp.text}")
        return []
    except Exception as e:
        gr.Warning(f"Connection error: {e}")
        return []

def upload_file_api(file_obj):
    if not file_obj:
        return None
    try:
        files = {"file": open(file_obj.name, "rb")}
        resp = requests.post(f"{API_URL}/upload", files=files)
        if resp.status_code == 200:
            return resp.json()["file_path"]
        gr.Warning(f"Upload failed: {resp.text}")
        return None
    except Exception as e:
        gr.Warning(f"Upload error: {e}")
        return None

def analyze_api(model, dataset, file_path, datasets_list, token, split=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {"model_repo_id": model, "split": split}
    
    # Construct datasets list from state + legacy inputs
    final_datasets = []
    if datasets_list:
        final_datasets.extend(datasets_list)
    
    # If explicit single inputs are provided (legacy or direct usage), add them too
    if file_path:
        final_datasets.append({"file_path": file_path, "split": split})
    elif dataset:
         final_datasets.append({"dataset_repo_id": dataset, "split": split})

    if final_datasets:
        payload["datasets"] = final_datasets
    else:
        return {"msg": "Please select a dataset or upload a file.", "needs_split": False}

    try:
        resp = requests.post(f"{API_URL}/analyze", json=payload, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            if "status" in data and data["status"] == "split_selection_needed":
                return {
                    "raw": data,
                    "msg": data["message"],
                    "splits": data["splits"],
                    "needs_split": True
                }
            else:
                return {
                    "raw": data,
                    "msg": "Analysis Successful!",
                    "needs_split": False
                }
        else:
            return {"msg": f"Error: {resp.text}", "needs_split": False}
    except Exception as e:
        return {"msg": f"Exception: {e}", "needs_split": False}

# ... (start_train_api, etc. remain same) ...

# UI
with gr.Blocks(title="Lightning Tune Pro") as app:
    gr.Markdown("# ⚡ Lightning Tune Pro")

    hf_token = gr.Textbox(label="Hugging Face Token", type="password")

    with gr.Tab("1. Setup"):
        with gr.Row():
            model_search = gr.Textbox(label="Search Model")
            model_dd = gr.Dropdown(label="Select Model")
            model_search.change(search_models_api, inputs=[model_search, hf_token], outputs=model_dd)

        gr.Markdown("### Dataset Setup")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Option A: Hugging Face")
                data_search = gr.Textbox(label="Search HF Dataset")
                data_dd = gr.Dropdown(label="Select HF Dataset")
                data_search.change(search_datasets_api, inputs=[data_search, hf_token], outputs=data_dd)
                data_config_name = gr.Textbox(label="Config Name (Optional)")
                data_split = gr.Textbox(label="Split (Optional)", placeholder="train")
                add_hf_btn = gr.Button("Add HF Dataset")

            with gr.Column(scale=1):
                gr.Markdown("#### Option B: Upload File")
                file_in = gr.File(label="Upload CSV/JSON")
                add_file_btn = gr.Button("Add Uploaded File")

        # Dataset List Display
        datasets_state = gr.State([])
        dataset_display = gr.Dataframe(
            headers=["Type", "ID/Path", "Config", "Split"], 
            datatype=["str", "str", "str", "str"],
            label="Selected Datasets",
            interactive=False
        )
        clear_ds_btn = gr.Button("Clear All Datasets", variant="secondary")

        # Internal states for upload
        uploaded_path = gr.State()

        # Upload handler
        def handle_upload(f):
            if f:
                path = upload_file_api(f)
                return path
            return None
        file_in.upload(handle_upload, inputs=[file_in], outputs=[uploaded_path])

        # Add buttons logic
        def add_hf_dataset(current_list, repo, conf, split):
            if not repo:
                 return current_list, gr.update()
            new_entry = {"repo_id": repo, "config_name": conf, "split": split, "type": "hf"}
            current_list.append(new_entry)
            display_data = [[d.get("type"), d.get("repo_id"), d.get("config_name"), d.get("split")] for d in current_list]
            return current_list, display_data

        def add_file_dataset(current_list, path):
            if not path:
                 gr.Warning("Please upload a file first.")
                 return current_list, gr.update()
            new_entry = {"file_path": path, "type": "file"}
            current_list.append(new_entry)
            display_data = [[d.get("type"), d.get("file_path"), "-", "-"] for d in current_list]
            return current_list, display_data
        
        def clear_datasets():
            return [], []

        add_hf_btn.click(add_hf_dataset, inputs=[datasets_state, data_dd, data_config_name, data_split], outputs=[datasets_state, dataset_display])
        add_file_btn.click(add_file_dataset, inputs=[datasets_state, uploaded_path], outputs=[datasets_state, dataset_display])
        clear_ds_btn.click(clear_datasets, outputs=[datasets_state, dataset_display])

        gr.Markdown("---")
        analyze_btn = gr.Button("Analyze All Datasets", variant="primary")
        
        # Legacy/Single split handling for analysis result fallback
        split_dd = gr.Dropdown(label="Select Split (for ambiguity resolution)", visible=False)
        analysis_status = gr.Textbox(label="Status", interactive=False)
        json_config = gr.JSON(label="Generated Config", visible=True)

        final_config = gr.State()

        def do_analyze(mod, d_list, tok, spl=None):
            # We treat d_list as the primary source now.
            # analyze_api expects (model, dataset, file_path, datasets_list, token, split)
            res = analyze_api(mod, None, None, d_list, tok, spl)
            
            if res.get("needs_split"):
                return {
                    analysis_status: res["msg"],
                    split_dd: gr.update(visible=True, choices=res["splits"]),
                    final_config: None,
                    json_config: None
                }
            else:
                return {
                    analysis_status: res["msg"],
                    split_dd: gr.update(visible=False),
                    final_config: res.get("raw"),
                    json_config: res.get("raw")
                }

        analyze_btn.click(do_analyze, inputs=[model_dd, datasets_state, hf_token, split_dd],
                          outputs=[analysis_status, split_dd, final_config, json_config])

    with gr.Tab("2. Train"):
        with gr.Row():
            push_check = gr.Checkbox(label="Push to Hub after training?")
            hub_model_id = gr.Textbox(label="Hub Model ID (username/repo)")

        with gr.Row():
            train_btn = gr.Button("Start Training", variant="primary")
            stop_btn = gr.Button("Stop Job", variant="stop")
        
        with gr.Row():
            job_id_display = gr.Textbox(label="Job ID", interactive=False)
            status_display = gr.Textbox(label="Job Status", interactive=False, value="Idle")

        logs_box = gr.Textbox(label="Live Logs", lines=20, max_lines=20, interactive=False)

        def start_training_wrapper(cfg, push, hid, tok):
            if not cfg:
                gr.Warning("No config generated! Please analyze a dataset first.")
                return None, "Failed (No Config)"
            jid = start_train_api(cfg, push, hid, tok)
            if jid:
                return jid, "Running"
            return None, "Failed (API Error)"

        train_btn.click(start_training_wrapper, inputs=[final_config, push_check, hub_model_id, hf_token],
                        outputs=[job_id_display, status_display])

        # Stop Button
        stop_btn.click(stop_job_api, inputs=[job_id_display, hf_token], outputs=[status_display])

        # Status Polling
        # Create a timer that updates status every 2 seconds if a job_id exists
        # Note: gr.Timer might not be in all versions, fallback to event loop? 
        # Assuming modern gradio (>4.0) has Timer.
        timer = gr.Timer(2)
        
        def poll_status(jid, tok):
            if not jid or "Failed" in str(jid):
                return "Idle"
            return get_job_status_api(jid, tok)

        timer.tick(poll_status, inputs=[job_id_display, hf_token], outputs=[status_display])

        async def stream_logs_gen(jid):
            if not jid:
                yield "Waiting for job..."
                return
            uri = f"{WS_URL}/train/{jid}/logs"
            try:
                async with websockets.connect(uri) as websocket:
                    logs = ""
                    async for message in websocket:
                        logs += message + "\n"
                        yield logs
            except Exception as e:
                yield f"Log stream disconnected: {e}"

        job_id_display.change(stream_logs_gen, inputs=[job_id_display], outputs=[logs_box])

        # Manual Push
        gr.Markdown("### Manual Push")
        manual_push_btn = gr.Button("Push Existing Job to Hub")
        push_status = gr.Textbox(label="Push Status")
        manual_push_btn.click(push_to_hub_api, inputs=[job_id_display, hub_model_id, gr.State(False), hf_token], outputs=push_status)

    with gr.Tab("3. Chat / Serve"):
        gr.Markdown("Serve the trained model and chat with it.")
        with gr.Row():
            serve_port = gr.Number(value=8000, label="Port")
            use_vllm_chk = gr.Checkbox(label="Use vLLM (Fast Inference)", value=True)
        serve_btn = gr.Button("Launch Server")
        service_url_display = gr.Textbox(label="Service URL")

        serve_btn.click(start_serve_api, inputs=[job_id_display, serve_port, use_vllm_chk, hf_token], outputs=service_url_display)

        chatbot = gr.Chatbot()
        with gr.Row():
             msg = gr.Textbox(label="Message")
             audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio Input (Optional)")
        clear = gr.Button("Clear")

        def chat_fn(message, audio_path, history, url):
            if not url:
                return history + [[message, "Server not connected."]]
            
            payload = {"prompt": message}
            if audio_path:
                try:
                    with open(audio_path, "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                    payload["audio_b64"] = audio_b64
                    message += " [Audio Attached]"
                except Exception as e:
                    return history + [[message, f"Error processing audio: {e}"]]

            try:
                resp = requests.post(f"{url}/predict", json=payload)
                if resp.status_code == 200:
                    reply = resp.json().get("completion", "No response")
                    history.append((message, reply))
                    return history
                else:
                    history.append((message, f"Error: {resp.text}"))
                    return history
            except Exception as e:
                history.append((message, f"Exception: {e}"))
                return history

        msg.submit(chat_fn, inputs=[msg, audio_input, chatbot, service_url_display], outputs=chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    app.launch(server_port=7860)
