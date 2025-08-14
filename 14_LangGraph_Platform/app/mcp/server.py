from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from app.mcp.tools.check_ip import AbuseipdbCheck

mcp = FastMCP("mcp-server")

@mcp.tool()
def check_ip(target: str, max_age: int = 30) -> str:
    """checks if the ip address is malicious"""
    abuseipdb = AbuseipdbCheck(target=target, max_age=max_age).check_status()
    return str(abuseipdb)

if __name__ == "__main__":
    mcp.run(transport="stdio")