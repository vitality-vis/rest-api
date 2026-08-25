"""Full-app paper export endpoints (not part of the papers profile)."""

from flask import Blueprint, Response, request
from flask_cors import cross_origin

from service import lib, zilliz


export_bp = Blueprint("export", __name__)


@export_bp.route("/checkoutPapers", methods=["POST"])
@cross_origin()
def checkout_papers():
    input_payload = request.json
    # Prioritize "input_data" field from frontend
    received_data = input_payload.get("input_data", [])

    paper_ids = []
    if received_data and isinstance(received_data[0], dict):
        paper_ids = [str(p.get("ID")) for p in received_data if p.get("ID") is not None]
    elif received_data:
        paper_ids = [str(pid) for pid in received_data]

    if not paper_ids:
        return Response("No valid paper IDs provided.", status=400)

    filename = "papers-checked-out.bibtex"
    papers = zilliz.query_doc_by_ids(paper_ids)
    response_text = "\n".join([lib.bib_template(paper) for paper in papers])
    return Response(
        response_text,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment;" + filename},
    )
