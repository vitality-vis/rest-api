import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

import json
import numpy as np
from datetime import datetime
from logger_config import setup_logger

# Initialize centralized logger (with Google Cloud Logging)
# Note: Import as custom_logger to avoid shadowing the logging module
from logger_config import get_logger
import logging as logging_module
logger = get_logger()
from flask import Flask, request, Response, jsonify
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit
from service.chain_chroma import chat_streaming_output, summarize_output, literature_review_output
from extension.ext_chroma import cached_data
from model.const import EMBED  
from model.chroma import ChromaQuerySchema
from service import chroma  
from service.chroma import query_doc_by_ids, query_docs, normalize_results
import config
# from service.agent_runner import build_agent, run_two_stage_hybrid_rag  # include new two-stage RAG
# from service.agent_runner import build_agent, run_two_stage_rag
from langchain_openai import AzureChatOpenAI
from prompt import SUMMARIZE_PROMPT, LITERATURE_REVIEW_PROMPT
from service import rag_core

class NoStopAzureChatOpenAI(AzureChatOpenAI):
    """AzureChatOpenAI wrapper that removes 'stop' for GPT-5 / Azure models."""
    def _generate(self, messages, stop=None, **kwargs):
        # Azure GPT-5 doesn’t allow 'stop' — remove it
        return super()._generate(messages, stop=None, **kwargs)

    def generate(self, messages, stop=None, **kwargs):
        return super().generate(messages, stop=None, **kwargs)

    def generate_prompt(self, prompts, stop=None, **kwargs):
        return super().generate_prompt(prompts, stop=None, **kwargs)


# === Initialize RAG Agent ===
_rag_agent = None
def get_rag_agent():
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = build_agent()
    return _rag_agent

# ===== Flask + SocketIO Init =====
app = Flask(__name__, static_folder='./build', static_url_path='/')
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
socketio = SocketIO(app, cors_allowed_origins=[
    'http://localhost:8080',  # User study dev server
    'http://localhost:8081', # standalone
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


##############insertion#######################
@socketio.on('log_event')
def handle_log_event(data):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        print("✅ PYTHON SERVER - Received log_event!")

        # Extract event details
        event_name = data.get("eventName", "unknown_event")
        user_id = data.get("userId")
        session_id = data.get("sessionId")
        study_id = data.get("studyId")
        event_data = data.get("eventData")

        # Log detailed information to TERMINAL and GCP (automatically sent to both)
        logger.info(
            f"[{timestamp}] Socket Event - "
            f"Event: {event_name} | "
            f"User ID: {user_id} | "
            f"Session ID: {session_id} | "
            f"Study ID: {study_id} | "
            f"Data: {json.dumps(event_data)}"
        )
    except KeyError as e:
        logger.error(f"[{timestamp}] Couldn't process the following: {e}")
        logger.info(f"Raw data received: {data}")
    except Exception as e:
        logger.error(f"[{timestamp}] An error occured during logging event: {e}")
        logger.info(f"Raw data received: {data}")




# # Initialize Flask app and enable CORS
# app = Flask(__name__, static_folder='./build', static_url_path='/')
# cors = CORS(app, resources={r"/*": {"origins": "*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'

# === Utility function to fetch similar papers using embedding and dimension (2D or nD) ===
def get_similarities_v2(paper_ids, embedding, dimensions, limit):
    if embedding not in EMBED.ALL:
        raise ValueError('Embedding must be one of {}'.format(EMBED.ALL))

    papers = chroma.query_doc_by_ids(paper_ids)
    if not papers:
        return []

    if dimensions == "nD":
        return chroma.query_similar_doc_by_embedding_full(papers, embedding, limit)
    elif dimensions == '2D':
        return chroma.query_similar_doc_by_embedding_2d(papers, embedding, limit)

# === Fallback embedding-based similarity query using paper abstract only ===
def get_similarities_by_abstract_v2(input_data, embedding, limit):
    # This function is not directly called by Flask routes, but may be used elsewhere
    # Modify according to embed_chroma convention if localization is needed
    return chroma.query_similar_doc_by_paper(input_data, embedding, limit)

# === Route: Retrieve papers based on filters (title, author, year, etc.) ===
@app.route('/getPapers', methods=['GET', 'POST'])
@cross_origin()
def get_papers():
    input_payload = request.args if request.method == 'GET' else request.json or {}

    query = ChromaQuerySchema(
        title=input_payload.get('title'),
        abstract=input_payload.get('abstract'),
        author=input_payload.get('author'),
        source=input_payload.get('source'),
        keyword=input_payload.get('keyword'),
        min_year=input_payload.get('min_year'),
        max_year=input_payload.get('max_year'),
        min_citation_counts=input_payload.get('min_citation_counts'),
        max_citation_counts=input_payload.get('max_citation_counts'),
        id_list=input_payload.get('id_list'),
        offset=int(input_payload.get('offset', 0)),
        limit=int(input_payload.get('limit', 1000))
    )
    return jsonify(query_docs(query))

@app.route('/getPapersLimited', methods=['POST'])
@cross_origin()
def get_papers_limited():
    input_payload = request.json or {}
    topic = input_payload.get("topic")
    limit = int(input_payload.get("limit", 10))
    offset = int(input_payload.get("offset", 0))

    if not topic:
        return jsonify({"message": "Topic is required"}), 400

    from service import embed_chroma as embed_service

    # Always use specter embedding
    query_embedding = embed_service.specter_embedding({
        "Title": topic,
        "Abstract": ""
    })

    # Query Chroma, results may be a list or a dict
    results = chroma.query_doc_by_embedding(
        paper_ids=[],
        embedding=query_embedding,
        embedding_type="specter",
        limit=limit * 5   # Fetch extra results to ensure enough candidates
    )

    # 🔍 Debug log: check type and structure of results
    print(f"[DEBUG] getPapersLimited result type: {type(results)}, "
          f"keys: {list(results.keys()) if isinstance(results, dict) else 'N/A'}, "
          f"length: {len(results) if isinstance(results, list) else 'N/A'}")

    # If results is a dict, slice only the list fields
    if isinstance(results, dict):
        for key in results:
            if isinstance(results[key], list):
                results[key] = results[key][offset: offset + limit]

    # If results is a list, slice directly
    elif isinstance(results, list):
        results = results[offset: offset + limit]

    return jsonify(results)

# === Route: Get similar papers based on abstract input and embedding model ===
# @app.route('/getSimilarPapersByAbstract', methods=['POST'])
@app.route('/getSimilarPapersByAbstract', methods=['POST'])
@cross_origin()
def get_similar_papers_by_abstract():
    import traceback
    input_payload = request.json
    abstract_text = input_payload.get("input_data", "")
    title_text = input_payload.get("title", "")

    embedding_type = input_payload.get("embedding", EMBED.SPECTER)
    limit = int(input_payload.get("limit", 25))
    query_lang = input_payload.get("lang", "all") 

    if not abstract_text:
        return jsonify({"message": "Abstract text is required"}), 400

    try:
        from service import embed_chroma as embed_service

        query_embedding = []
        # if embedding_type == EMBED.ADA:
        #     query_embedding = embed_service.ada_embedding({'abstract': abstract_text})
        # elif embedding_type == EMBED.GLOVE:
        #     query_embedding = embed_service.glove_embedding({'text': abstract_text})
        if embedding_type == EMBED.ADA:
            query_embedding = embed_service.ada_embedding(abstract_text)
        elif embedding_type == EMBED.GLOVE:
            query_embedding = embed_service.glove_embedding(abstract_text)
        elif embedding_type == EMBED.SPECTER:
            query_embedding = embed_service.specter_embedding({'Title': title_text, 'Abstract': abstract_text})
        else:
            logger.error(f"Unsupported embedding type: {embedding_type}")
            return jsonify({"message": f"Unsupported embedding type: {embedding_type}"}), 400

        # Handle nested embeddings safely
        if isinstance(query_embedding, (list, np.ndarray)) and len(query_embedding) > 0:
            first = query_embedding[0]
            if isinstance(first, (list, np.ndarray)) and not isinstance(first, str):
                query_embedding = first

        if not query_embedding or not isinstance(query_embedding, (list, np.ndarray)):
            logger.error(f"Invalid or empty embedding: {query_embedding}")
            return jsonify({"message": "Failed to generate embedding"}), 500

        query_embedding_np = np.array(query_embedding)
        query_embedding_norm = np.linalg.norm(query_embedding_np)
        logger.debug(f"Real-time query_embedding L2 Norm: {query_embedding_norm:.6f}")
        logger.debug(f"[Embedding] Final query embedding length: {len(query_embedding)}")

        language_filter = {"lang": query_lang} if query_lang and query_lang != "all" else {}

        results = chroma.query_doc_by_embedding(
            paper_ids=[],
            embedding=query_embedding,
            embedding_type=embedding_type,
            limit=limit,
            lang_filter=language_filter
        )

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in get_similar_papers_by_abstract: {e}")
        traceback.print_exc()
        return jsonify({"message": f"Internal server error: {e}"}), 500

@app.route('/getSimilarPapers', methods=['POST'])
@cross_origin()
def get_similar_papers():
    try:
        input_payload = request.json
        if isinstance(input_payload, str):
            try:
                input_payload = json.loads(input_payload)
            except json.JSONDecodeError:
                return jsonify({"message": "Request body is a string but not valid JSON."}), 400

        papers_data = input_payload.get("input_data", [])
        embedding_type = input_payload.get("embedding", EMBED.SPECTER)
        limit = int(input_payload.get("limit", 25))
        query_lang = input_payload.get("lang", "all")
        dimensions = input_payload.get("dimensions", "nD")  

        # if not papers_data:
        #     return jsonify({"message": "A list of papers or paper IDs is required"}), 400


        # if papers_data and isinstance(papers_data[0], str):
        #     papers = chroma.query_doc_by_ids(papers_data)
        # else:
        #     papers = papers_data

##############insertion###################
        # --- THIS IS THE NEW LOGIC ---
        # Check if we received a list of IDs (strings) or full paper objects (dicts)
        if papers_data and isinstance(papers_data[0], str):
            # It's a list of IDs, so fetch the full paper objects from the database
            papers = chroma.query_doc_by_ids(papers_data)
        else:
            # It's already a list of full paper objects
            papers = papers_data
        # ---------------------------

        if not papers:
            return jsonify({"message": "Could not find details for the provided paper IDs"}), 404

        language_filter = {"lang": query_lang} if query_lang and query_lang != "all" else None

        if dimensions == "2D":
            raw_results = chroma.query_similar_doc_by_embedding_2d(
                papers=papers,
                embedding_type=embedding_type,
                limit=limit,
                lang_filter=language_filter
            )
            results = normalize_results(raw_results, mode="2D")
        else:  
            raw_results = chroma.query_similar_doc_by_embedding_full(
                papers=papers,
                embedding_type=embedding_type,
                limit=limit,
                lang_filter=language_filter
            )
            results = normalize_results(raw_results, mode="nD")

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in get_similar_papers: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"message": f"Internal server error: {e}"}), 500

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
    papers = chroma.query_doc_by_ids(paper_ids)
    from service import lib
    response_text = '\n'.join([lib.bib_template(paper) for paper in papers])
    return Response(response_text, mimetype="text/plain", headers={"Content-Disposition": "attachment;" + filename})


from service.agent_runner import run_two_stage_rag_stream
import asyncio
from flask import Response, request
from flask_cors import cross_origin


@app.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    """
    Stable streaming endpoint
    --------------------------------
    ✔ No event-loop destruction
    ✔ No 'Task was destroyed but it is pending!'
    ✔ No 'coroutine was never awaited'
    ✔ Works with async run_two_stage_rag_stream()
    """

    data = request.get_json(force=True) or {}
    text = data.get('text', '').strip()
    chat_id = data.get('chat_id', 'default')

    if not text:
        return Response("Please Input Your Text", status=400)

    # ---- Create ONE event loop for this request ----
    loop = asyncio.new_event_loop()

    # Run async generator inside this loop
    async def agen():
        async for chunk in run_two_stage_rag_stream(text, chat_id):
            yield chunk

    # Sync wrapper
    def stream_sync():
        try:
            agen_obj = agen().__aiter__()
            while True:
                chunk = loop.run_until_complete(agen_obj.__anext__())
                yield chunk
        except StopAsyncIteration:
            pass
        finally:
            # Don't close the loop too early!
            # Let background tasks finish
            pending = asyncio.all_tasks(loop=loop)
            for task in pending:
                task.cancel()
            try:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except:
                pass
            loop.close()

    return Response(
        stream_sync(),
        status=200,
        mimetype="text/plain"
    )

from flask import Response, request
from flask_cors import cross_origin
# from service.agent_runner import streaming_llm  
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






# === Route: Retrieve all UMAP projection points for visualization ===
@app.route('/getUmapPoints', methods=['GET'])
@cross_origin()
def get_umap_points():
    return jsonify(cached_data.get_umap_points())

# === Route: Get metadata aggregates for filters ===
@app.route('/getMetaData', methods=['GET'])
@cross_origin()
def get_metas():
    # Use cached metadata instead of computing it every time
    cached_metadata = cached_data.get_aggregated_metadata()
    if cached_metadata:
        return jsonify(cached_metadata)

    # Fallback to real-time computation if cache is not available
    logger.warning("⚠️ Metadata cache not available, computing in real-time (slow)")
    return jsonify({
        'authors_summary': chroma.get_distinct_authors_with_counts(),
        'sources_summary': chroma.get_distinct_sources_with_counts(),
        'keywords_summary': chroma.get_distinct_keywords_with_counts(),
        'years_summary': chroma.get_distinct_years_with_counts(),
        'titles': chroma.get_distinct_titles(),
        'citation_counts': chroma.get_distinct_citation_counts()
    })

# ... (在 LLM 初始化之后，路由定义之前)




llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    streaming=True  
)


def format_papers_in_prompt(papers):
    lines = []
    for p in papers:
        title = p.get('Title', '')
        authors_data = p.get('Authors', [])
        authors = ', '.join(authors_data) if isinstance(authors_data, list) else str(authors_data)
        
        abstract = p.get('Abstract', '')
        source = p.get('Source', '')
        year = p.get('Year', '')
        
        keywords_data = p.get('Keywords', [])
        keywords = ', '.join(keywords_data) if isinstance(keywords_data, list) else str(keywords_data)

        lines.append(
            f" --- \nTitle: {title}\nAuthors: {authors}\nAbstract: {abstract}\nSource: {source}\nYear: {year}\nKeywords: {keywords}\n ---"
        )
    return "\n".join(lines)

def summarize_output(prompt_data):
    return (i.content for i in llm.stream(SUMMARIZE_PROMPT.format(**prompt_data)))

def literature_review_output(prompt_data):
    return (i.content for i in llm.stream(LITERATURE_REVIEW_PROMPT.format(**prompt_data)))
    
@app.route('/summarize', methods=['POST'])
@cross_origin()
def summarize():
    try:
        data = request.json or {}
        prompt = data.get('prompt', '')
        paper_ids = data.get('ids', [])

        if not paper_ids:
            return Response("Error: Saved paper list is empty", status=400)

        selected_papers = chroma.query_doc_by_ids(paper_ids)
        if not selected_papers:
            return Response("Error: No papers found for the given IDs", status=404)

        formatted_content = format_papers_in_prompt(selected_papers)
        output_generator = summarize_output({
            'prompt': prompt,
            'content': formatted_content
        })
        

        return Response(output_generator, mimetype='text/plain')

    except Exception as e:
        traceback.print_exc()
        return Response(f"An internal error occurred: {str(e)}", status=500)


@app.route('/literatureReview', methods=['POST'])
@cross_origin()
def literature_review():
    try:
        data = request.json or {}
        prompt = data.get('prompt', '')
        paper_ids = data.get('ids', [])

        if not paper_ids:
            return Response("Error: Saved paper list is empty", status=400)

        selected_papers = chroma.query_doc_by_ids(paper_ids)
        if not selected_papers:
            return Response("Error: No papers found for the given IDs", status=404)

        formatted_content = format_papers_in_prompt(selected_papers)
        
        # 直接获取 LLM 的输出生成器
        output_generator = literature_review_output({
            'prompt': prompt,
            'content': formatted_content
        })

        # 同样，直接返回流式响应
        return Response(output_generator, mimetype='text/plain')

    except Exception as e:
        traceback.print_exc()
        return Response(f"An internal error occurred: {str(e)}", status=500)


# === Route: Serve frontend index.html ===
@app.route('/')
@cross_origin()
def index():
    return app.send_static_file('index.html')

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

    papers = chroma.query_doc_by_ids([paper_id])
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

    papers = chroma.query_doc_by_title(title)
    return jsonify(papers)


from service.agent_runner import reset_all_sessions

# 🧹 On startup
reset_all_sessions()
print("[startup] 🔄 Cleared all chat sessions (docs + memory).")

@app.route("/resetMemory", methods=["POST"])
@cross_origin()
def reset_memory():
    try:
        reset_all_sessions()
        print("[resetMemory] 🧹 Cleared all sessions (docs + chat memory).")
        return jsonify({"status": "success", "message": "All sessions cleared."})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

cached_data.init()


# === Start the Flask-SocketIO server ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    #cached_data.init()
    print(f"🚀 Starting Flask-SocketIO server on http://localhost:{port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
