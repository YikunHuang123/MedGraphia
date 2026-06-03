"""
Streamlit frontend (Phase 9).

Entry point: `streamlit_app.py`. Multipage layout under `pages/`:
    1_Chat.py            — SSE-streamed Q&A with citation expansion
    2_Graph_Explorer.py  — entity auto-complete + pyvis subgraph render
    3_Dashboard.py       — backend health + graph statistics
    4_Admin.py           — pipeline trigger/monitor + API-key lifecycle

Run locally:
    pip install -e ".[ui]"
    streamlit run src/medgraphia/ui/streamlit_app.py
"""
