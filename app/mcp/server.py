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
            "Use BM25 when matching supplied terminology matters, semantic "
            "search for conceptual questions, and filter_papers for metadata-only "
            "requests. All tools are public and read-only."
        ),
    )
    register_public_tools(server)
    return server
