import os, json, re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import gradio as gr

# Import prompt configuration module
from config_prompts import get_financial_prompt

# 1. Global Initialization
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    db = MilvusClient("./financial_rag.db")
except Exception as e:
    print(f"Init Error: {e}")
    model, db = None, None

app = FastAPI()

class NewsInput(BaseModel):
    news_text: str

# 2. Core Analysis Function (called by both API and UI)
def run_core_analysis(news_text: str):
    core_news = news_text[:1000]

    # RAG retrieval
    retrieved_context = "No specific rules found in the current knowledge base."
    if db and model:
        try:
            query_vec = model.encode([core_news])[0].tolist()
            search_res = db.search(collection_name="finance_rules", data=[query_vec], limit=1, output_fields=["text"])
            if search_res and len(search_res[0]) > 0:
                retrieved_context = search_res[0][0]['entity'].get('text', retrieved_context)
        except Exception:
            pass

    system_prompt = get_financial_prompt(retrieved_context)

    # Call the LLM
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"News: {core_news}\nResult:"}
        ]
    )

    think = getattr(response.choices[0].message, 'reasoning_content', "")
    full_content = response.choices[0].message.content

    # Parse JSON
    json_match = re.search(r'\{.*\}', full_content, re.DOTALL)
    signal = json.loads(json_match.group().replace("'", '"')) if json_match else {"direction": 0, "confidence": 0, "note": "Parse error"}

    return {
        "final_signal": signal,
        "think_process": think if think else "No reasoning process captured.",
        "context": retrieved_context,
        "dehydrated_news": core_news
    }

# 3. FastAPI Route (for programmatic access)
@app.post("/api/v1/analyze")
def analyze_api(input_data: NewsInput):
    try:
        data = run_core_analysis(input_data.news_text)
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. Gradio Web UI (for human access)
def analyze_ui(news_text):
    if not news_text or not news_text.strip():
        return "⚖️ Neutral (0)", "0%", "N/A", "Please enter financial news.", "N/A"
    try:
        # The web page directly calls the core function, no network request needed, faster!
        data = run_core_analysis(news_text)

        sig = data.get("final_signal", {})
        direction = sig.get("direction", 0)

        dir_str = "📈 Bullish (1)" if direction == 1 else "📉 Bearish (-1)" if direction == -1 else "⚖️ Neutral (0)"
        conf_str = f"{sig.get('confidence', 0)}%"

        return dir_str, conf_str, data.get("context"), data.get("think_process"), data.get("dehydrated_news")
    except Exception as e:
        return "❌ Error", "0%", "N/A", f"Processing Error: {str(e)}", "N/A"

with gr.Blocks(theme=gr.themes.Soft(), title="DeepSeek Quant Terminal") as demo:
    gr.Markdown(
        """
        <div style='text-align: center;'>
        <h1>🚀 FinGPT & DeepSeek Quantitative Signal Terminal</h1>
        <p>Backend Status: <b>Cloud Run (us-east4) Active</b> | Engine: <b>DeepSeek-R1-Reasoner</b></p>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 🎛️ Signal Control Panel")
            news_input = gr.Textbox(label="📰 Enter Financial News Article", placeholder="Paste news here...", lines=10)
            analyze_btn = gr.Button("🚀 Execute DeepSeek Deep Inference", variant="primary")
            gr.Markdown("---")
            gr.Examples(
                examples=[
                    ["NVIDIA reports record revenue, beating expectations with strong AI chip demand."],
                    ["The company announced a potential bankruptcy filing following a failed debt restructuring."],
                    ["Market remains steady as the Federal Reserve keeps interest rates unchanged."]
                ],
                inputs=[news_input]
            )
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Real-time Quantitative Analysis")
            with gr.Row():
                direction_out = gr.Label(label="Predicted Direction")
                confidence_out = gr.Textbox(label="🎯 Signal Confidence", scale=1)
            with gr.Accordion("📚 Knowledge Base Reference (RAG Context)", open=False):
                rag_out = gr.Textbox(label="Hit Financial Audit Rules", lines=3)
            with gr.Accordion("🧠 DeepSeek Internal Logic Chain of Thought (XAI)", open=True):
                cot_out = gr.Textbox(label="Reasoning Process (Chain of Thought)", lines=14)
            with gr.Accordion("💧 LLM Information Dehydration Result", open=False):
                dehydrated_out = gr.Textbox(label="Cleaned facts", lines=4)

    analyze_btn.click(
        fn=analyze_ui,
        inputs=[news_input],
        outputs=[direction_out, confidence_out, rag_out, cot_out, dehydrated_out]
    )

# 🌟 Mount Gradio at the root directory "/" of FastAPI
app = gr.mount_gradio_app(app, demo, path="/")
