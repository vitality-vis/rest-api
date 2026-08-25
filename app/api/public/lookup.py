"""Public single-paper lookup endpoints."""

from flask import Blueprint, jsonify, request
from flask_cors import cross_origin

from service import zilliz


lookup_bp = Blueprint("paper_lookup", __name__)


@lookup_bp.route("/getPaperById", methods=["GET"])
@cross_origin()
def get_paper_by_id():
    paper_id = request.args.get("id")
    if not paper_id:
        return jsonify({"message": "No ID provided"}), 400

    papers = zilliz.query_doc_by_ids([paper_id])
    if papers and len(papers) > 0:
        return jsonify(papers[0])
    return jsonify({})


@lookup_bp.route("/getPaperByTitle", methods=["POST"])
@cross_origin()
def get_paper_by_title():
    data = request.json or {}
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"message": "No title provided"}), 400

    papers = zilliz.query_doc_by_title(title)
    return jsonify(papers)
