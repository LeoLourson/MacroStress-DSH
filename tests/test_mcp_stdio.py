"""The server answers over its real stdio transport, where loading plugin code can block.

On Windows the first import of scipy's compiled extensions inside a running server never
returned while the transport's thread waited on stdin, so the first tool that loaded the
plugins hung until the client closed the pipe.
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ANSWER_SECONDS = 20


def test_the_first_call_that_loads_the_plugins_answers_over_stdio(tmp_path):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mst.mcp", "--workspace", str(tmp_path)],
        env={**os.environ, "PYTHONUTF8": "1"},
    )

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                call = session.call_tool("model", {"kind": "catalog", "action": "list"})
                return await asyncio.wait_for(call, ANSWER_SECONDS)

    result = asyncio.run(run())
    assert not result.isError
    assert "bvar" in result.content[0].text
