import os
from datetime import datetime
import argparse

# Load .env before Google Cloud Logging (needs GOOGLE_APPLICATION_CREDENTIALS).
import config
config.load_project_environment()

# Initialize PyMilvus before attaching the Google Cloud logging handler.
from service.bootstrap import initialize_runtime
import logging as logging_module
logger = initialize_runtime()
from flask import Flask, request, Response, jsonify
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit
from flask_compress import Compress
from service.static_cache import cached_data
from app.api.bootstrap import bootstrap_bp
from app.api.chat import chat_bp
from app.api.papers import papers_bp
from app.api.library import library_bp
from app.api.notes import notes_bp
from agents.agent_v1_legacy.summary_routes import legacy_summary_bp
from service import zilliz

# === Initialize RAG Agent ===
_rag_agent = None
def get_rag_agent():
    """Return the RAG agent. Agent is created per-session in agent_runner.get_or_create_chat_session()."""
    return None  # Use agent_runner.get_or_create_chat_session(chat_id)["agent"] for chat agent

# ===== Flask + SocketIO Init =====
app = Flask(
    __name__,
    static_folder=config.FRONTEND_DIST_DIR,
    static_url_path="/",
)
app.register_blueprint(bootstrap_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(papers_bp)
app.register_blueprint(library_bp)
app.register_blueprint(notes_bp)
app.register_blueprint(legacy_summary_bp)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type, Authorization'

# Enable Gzip compression for all responses (reduces JSON payload by ~80%)
Compress(app)

# socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins (for tunnel access)
socketio = SocketIO(app, cors_allowed_origins=[
    'http://localhost:8080',  # User study dev server
    'http://localhost:8081', # standalone
    'http://localhost:5173', # rebuild Vite dev server
    'https://vitality.mathcs.emory.edu'  # Production  server
])

# Configure Flask's logger to work with our custom logger
app.logger.handlers = logger.handlers
app.logger.setLevel(logger.level)

# Reduce SocketIO/engineio logging noise (set to WARNING to only show important messages)
logging_module.getLogger('socketio').setLevel(logging_module.WARNING)
logging_module.getLogger('engineio').setLevel(logging_module.WARNING)

# ===== SocketIO Event Handlers =====
@socketio.on('connect')
def handle_connect(auth):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'[{timestamp}] WebSocket Client connected: {request.sid}')
    emit('connected', {'data': 'Connected to Flask-SocketIO server'})

@socketio.on('disconnect')
def handle_disconnect():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'[{timestamp}] WebSocket Client disconnected: {request.sid}')


@socketio.on('log_event')
def handle_log_event(data):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        if not isinstance(data, dict):
            raise ValueError("event must be an object")

        # The rebuild sends a single provenance envelope. Keep this Socket.IO
        # channel and its Google Cloud Logging destination compatible with the
        # legacy study logger.
        event_id = data.get("eventId")
        session_id = data.get("sessionId")
        action = data.get("action")
        event_data = data.get("eventData")
        if not all(isinstance(value, str) and value for value in (event_id, session_id, action)):
            raise ValueError("eventId, sessionId, and action are required")
        if not isinstance(event_data, dict):
            raise ValueError("eventData must be an object")

        # Log detailed information to TERMINAL and GCP (automatically sent to both)
        overview = f"Socket Event - Actor Type: {data.get('actorType', 'unknown')} | Action: {action}"
        # CloudLoggingHandler recognises a mapping message and writes it
        # directly as jsonPayload.
        logger.info({"message": overview, **data}, extra={"provenance_event": True})

        # Returning from a Flask-SocketIO handler is the acknowledgement sent
        # to the callback passed to socket.emit on the client.
        return {"status": "success", "timestamp": timestamp}

    except Exception as e:
        logger.error(f"[{timestamp}] An error occured during logging event: {e}")
        logger.info(f"Raw data received: {data}")
        return {"status": "error", "message": str(e)}

# === Route: Download selected papers in BibTeX format ===
@app.route('/checkoutPapers', methods=['POST'])
@cross_origin()
def checkout_papers():
    input_payload = request.json
    # Prioritize "input_data" field from frontend
    received_data = input_payload.get('input_data', [])

    paper_ids = []
    if received_data and isinstance(received_data[0], dict):
        paper_ids = [str(p.get('ID')) for p in received_data if p.get('ID') is not None]
    elif received_data:
        paper_ids = [str(pid) for pid in received_data]

    if not paper_ids:
        return Response("No valid paper IDs provided.", status=400)

    filename = "papers-checked-out.bibtex"
    papers = zilliz.query_doc_by_ids(paper_ids)
    from service import lib
    response_text = '\n'.join([lib.bib_template(paper) for paper in papers])
    return Response(response_text, mimetype="text/plain", headers={"Content-Disposition": "attachment;" + filename})


from flask import Response, request
from flask_cors import cross_origin
# from agents.agent_v1_legacy.runner import streaming_llm  # legacy; undefined  
import asyncio


@app.route("/chat_stream_simple", methods=["POST"])
@cross_origin()
def chat_stream_simple():
    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return Response("Please Input Your Text", status=400)

    async def llm_stream():
        # single LLM call, token streaming
        async for chunk in streaming_llm.astream(text):
            # chunk is a ChatMessageChunk – its content is the new tokens
            yield chunk.content or ""

        # tell frontend we’re done
        yield "[[STREAM_DONE]]"

    def sync_stream():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            agen = llm_stream()
            while True:
                part = loop.run_until_complete(agen.__anext__())
                if not part:
                    continue
                yield part
        except StopAsyncIteration:
            pass
        finally:
            loop.close()

    return Response(sync_stream(), mimetype="text/plain", status=200)


# === Route: Serve frontend index.html ===
@app.route('/')
@cross_origin()
def index():
    return app.send_static_file('index.html')


@app.errorhandler(404)
def spa_fallback(error):
    """Serve the SPA shell for unknown GET paths (client-side routes)."""
    if request.method == "GET":
        static_path = os.path.join(app.static_folder or "", request.path.lstrip("/"))
        if request.path != "/" and os.path.isfile(static_path):
            return app.send_static_file(request.path.lstrip("/"))
        accept = request.accept_mimetypes
        if accept.accept_html or "text/html" in str(request.accept_mimetypes):
            return app.send_static_file("index.html")
    return jsonify({"message": "Not found"}), 404


# === Start the Flask server ===
# if __name__ == "__main__":
#     port = int(os.environ.get('PORT', 3000))
#     cached_data.init()
#     app.run(host='0.0.0.0', port=port)

@app.route("/getPaperById", methods=["GET"])
@cross_origin()
def get_paper_by_id():
    paper_id = request.args.get("id")
    if not paper_id:
        return jsonify({"message": "No ID provided"}), 400

    papers = zilliz.query_doc_by_ids([paper_id])
    if papers and len(papers) > 0:
        return jsonify(papers[0]) 
    return jsonify({})


@app.route("/getPaperByTitle", methods=["POST"])
@cross_origin()
def get_paper_by_title():
    data = request.json or {}
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"message": "No title provided"}), 400

    papers = zilliz.query_doc_by_title(title)
    return jsonify(papers)


from agents.agent_v1_legacy.runner import reset_all_sessions

# On startup
reset_all_sessions()
print("[startup] Cleared all chat sessions (docs + memory).")

@app.route("/resetMemory", methods=["POST"])
@cross_origin()
def reset_memory():
    try:
        reset_all_sessions()
        print("[resetMemory] Cleared all sessions (docs + chat memory).")
        return jsonify({"status": "success", "message": "All sessions cleared."})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

cached_data.init()


# === Start the Flask-SocketIO server ===
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start Flask-SocketIO server')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode (default: False)')
    args = parser.parse_args()

    port = int(os.environ.get("PORT", 3000))
    #cached_data.init()

    debug_mode = args.debug
    print(f"Starting Flask-SocketIO server on http://localhost:{port}")
    print(f"Debug mode: {debug_mode}")

    socketio.run(app, host="0.0.0.0", port=port, debug=debug_mode, use_reloader=debug_mode)
