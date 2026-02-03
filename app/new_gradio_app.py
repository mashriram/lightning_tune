import gradio as gr
import requests
import json
import websockets
import asyncio
import threading
from typing import Optional

API_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000"

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
        return []
    except:
        return []

def search_datasets_api(query, token):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(f"{API_URL}/datasets", params={"query": query, "limit": 10}, headers=headers)
        if resp.status_code == 200:
            return [d["id"] for d in resp.json()]
        return []
    except:
        return []

def upload_file_api(file_obj):
    if not file_obj:
        return None
    try:
        files = {"file": open(file_obj.name, "rb")}
        resp = requests.post(f"{API_URL}/upload", files=files)
        if resp.status_code == 200:
            return resp.json()["file_path"]
        return None
    except:
        return None

def analyze_api(model, dataset, file_path, token, split=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    # Prioritize file if present
    payload = {"model_repo_id": model, "split": split}
    if file_path:
        payload["file_path"] = file_path
    elif dataset:
        payload["dataset_repo_id"] = dataset
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

def start_train_api(config, push, hub_id, token):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {"config": config, "push_to_hub": push, "hub_model_id": hub_id}
    try:
        resp = requests.post(f"{API_URL}/train", json=payload, headers=headers)
        if resp.status_code == 200:
            return resp.json()["job_id"]
        return None
    except:
        return None

def start_serve_api(job_id, port, token):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {"job_id": job_id, "port": int(port)}
    try:
        resp = requests.post(f"{API_URL}/serve", json=payload, headers=headers)
        if resp.status_code == 200:
            return resp.json()["service_url"]
        return None
    except:
        return None

def push_to_hub_api(job_id, hub_id, private, token):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {"job_id": job_id, "hub_model_id": hub_id, "private": private}
    try:
        resp = requests.post(f"{API_URL}/push_to_hub", json=payload, headers=headers)
        if resp.status_code == 200:
            return resp.json()["message"]
        return f"Error: {resp.text}"
    except Exception as e:
        return f"Exception: {e}"

# UI
with gr.Blocks(title="Lightning Tune Pro") as app:
    gr.Markdown("# ⚡ Lightning Tune Pro")

    hf_token = gr.Textbox(label="Hugging Face Token", type="password")

    with gr.Tab("1. Setup"):
        with gr.Row():
            model_search = gr.Textbox(label="Search Model")
            model_dd = gr.Dropdown(label="Select Model")
            model_search.change(search_models_api, inputs=[model_search, hf_token], outputs=model_dd)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Option A: Hugging Face Dataset")
                data_search = gr.Textbox(label="Search Dataset")
                data_dd = gr.Dropdown(label="Select Dataset")
                data_search.change(search_datasets_api, inputs=[data_search, hf_token], outputs=data_dd)
            with gr.Column():
                gr.Markdown("### Option B: Upload File")
                file_in = gr.File(label="Upload CSV/JSON")

        analyze_btn = gr.Button("Analyze", variant="primary")
        split_dd = gr.Dropdown(label="Select Split", visible=False)
        analysis_status = gr.Textbox(label="Status", interactive=False)
        json_config = gr.JSON(label="Generated Config", visible=True)

        # Internal states
        final_config = gr.State()
        uploaded_path = gr.State()

        # Handle file upload automatically when file changes
        def handle_upload(f):
            if f:
                path = upload_file_api(f)
                return path
            return None

        file_in.upload(handle_upload, inputs=[file_in], outputs=[uploaded_path])

        def do_analyze(mod, dat, up_path, tok, spl=None):
            # Pass uploaded path if exists, otherwise dataset
            # If both, logic in analyze_api prioritizes file
            res = analyze_api(mod, dat, up_path, tok, spl)
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

        analyze_btn.click(do_analyze, inputs=[model_dd, data_dd, uploaded_path, hf_token, split_dd],
                          outputs=[analysis_status, split_dd, final_config, json_config])

    with gr.Tab("2. Train"):
        with gr.Row():
            push_check = gr.Checkbox(label="Push to Hub after training?")
            hub_model_id = gr.Textbox(label="Hub Model ID (username/repo)")

        train_btn = gr.Button("Start Training", variant="primary")
        job_id_display = gr.Textbox(label="Job ID", interactive=False)

        logs_box = gr.Textbox(label="Live Logs", lines=20, max_lines=20, interactive=False)

        def start_training_wrapper(cfg, push, hid, tok):
            if not cfg:
                return "No config generated!", ""
            jid = start_train_api(cfg, push, hid, tok)
            if jid:
                return jid, "Job started..."
            return "Failed", "Failed to start"

        train_btn.click(start_training_wrapper, inputs=[final_config, push_check, hub_model_id, hf_token],
                        outputs=[job_id_display, logs_box])

        async def stream_logs_gen(jid):
            if not jid or "Failed" in jid:
                yield "No job."
                return
            uri = f"{WS_URL}/train/{jid}/logs"
            try:
                async with websockets.connect(uri) as websocket:
                    logs = ""
                    async for message in websocket:
                        logs += message + "\n"
                        yield logs
            except Exception as e:
                yield f"Connection closed or error: {e}"

        job_id_display.change(stream_logs_gen, inputs=[job_id_display], outputs=[logs_box])

        # Manual Push
        gr.Markdown("### Manual Push")
        manual_push_btn = gr.Button("Push Existing Job to Hub")
        push_status = gr.Textbox(label="Push Status")
        manual_push_btn.click(push_to_hub_api, inputs=[job_id_display, hub_model_id, gr.State(False), hf_token], outputs=push_status)

    with gr.Tab("3. Chat / Serve"):
        gr.Markdown("Serve the trained model and chat with it.")
        serve_port = gr.Number(value=8000, label="Port")
        serve_btn = gr.Button("Launch Server")
        service_url_display = gr.Textbox(label="Service URL")

        serve_btn.click(start_serve_api, inputs=[job_id_display, serve_port, hf_token], outputs=service_url_display)

        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Message")
        clear = gr.Button("Clear")

        def chat_fn(message, history, url):
            if not url:
                return history + [[message, "Server not connected."]]
            try:
                resp = requests.post(f"{url}/predict", json={"prompt": message})
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

        msg.submit(chat_fn, inputs=[msg, chatbot, service_url_display], outputs=chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    app.launch(server_port=7860)
