"""MCP Server providing database tools backed by SQLite."""
import json
import os
import sqlite3

from mcp.server.fastmcp import FastMCP

DB_PATH = os.path.join(os.path.dirname(__file__), "sample.db")


def _ensure_db():
    """Create tables and seed data if the database does not exist."""
    if os.path.exists(DB_PATH):
        return
    from init_db import create_database
    create_database()


_ensure_db()

mcp = FastMCP("DB Server")


def _get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_tables() -> str:
    """List all table names in the database as a JSON array."""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor.fetchall()]
        return json.dumps(tables)
    finally:
        conn.close()


@mcp.tool()
def describe_table(table: str) -> str:
    """Show column information for a table as a JSON array of {name, type}.

    Args:
        table: The table name to describe.
    """
    conn = _get_conn()
    try:
        # PRAGMA 不支持参数绑定，转义双引号防注入
        safe_table = table.replace('"', '""')
        cursor = conn.execute(f'PRAGMA table_info("{safe_table}")')
        columns = [{"name": row["name"], "type": row["type"]} for row in cursor.fetchall()]
        return json.dumps(columns)
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        conn.close()


@mcp.tool()
def query_db(sql: str) -> str:
    """Execute a SELECT query and return results as a JSON array of rows.

    Only SELECT queries are allowed. Maximum 100 rows returned.
    """
    stripped = sql.strip().lower()
    if not stripped.startswith("select"):
        return "Only SELECT queries are allowed."

    conn = _get_conn()
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(100)
        results = [dict(row) for row in rows]
        return json.dumps(results, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8004"))
    import uvicorn
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
