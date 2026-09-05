"""Factory for the public papers MCP server."""

from mcp.server.mcpserver import MCPServer

from app.mcp.tools import register_public_tools


def create_public_mcp_server() -> MCPServer:
    server = MCPServer(
        name="vitality-papers",
        title="VitaLITy Paper Search",
        version="0.1.0",
        instructions=(
            "Search and inspect the public VitaLITy academic-paper corpus. "
            "Use Boolean search for combinations of exact words and quoted "
            "phrases, BM25 when lexical relevance matters, semantic search for "
            "conceptual questions, and filter_papers for metadata-only requests. "
            "All tools are public and read-only."
        ),
    )
    register_public_tools(server)
    return server
