import gradio as gr
import requests

API_URL = "http://localhost:8000/search"  # Update for Vercel deployment

def chat_search(query, party, constituency, policy_area):
    payload = {
        "query": query,
        "party": party if party != "Any" else None,
        "constituency": constituency if constituency != "Any" else None,
        "policy_area": policy_area if policy_area != "Any" else None,
        "top_k": 5
    }
    r = requests.post(API_URL, json=payload)
    if r.status_code == 200:
        data = r.json()
        # Format results for display
        return str(data)
    else:
        return "Error: " + r.text

# Dummy filter options for demo; replace with dynamic fetch from DB
parties = ["Any", "Workers' Party", "PAP"]
constituencies = ["Any", "West Region", "East Region"]
policy_areas = ["Any", "Affordable Housing", "Healthcare"]

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🇸🇬 Election Info Chat & Search")
        with gr.Row():
            query = gr.Textbox(label="Ask about candidates, parties, policies...")
        with gr.Row():
            party = gr.Dropdown(parties, label="Filter by Party")
            constituency = gr.Dropdown(constituencies, label="Filter by Constituency")
            policy_area = gr.Dropdown(policy_areas, label="Filter by Policy Area")
        chat = gr.Chatbot()
        submit = gr.Button("Search")
        def respond_fn(q, p, c, pa, history):
            answer = chat_search(q, p, c, pa)
            history = history or []
            history.append((q, answer))
            return "", history
        submit.click(respond_fn, [query, party, constituency, policy_area, chat], [query, chat])
    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
