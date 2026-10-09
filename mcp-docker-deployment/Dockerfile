FROM python:3.14-slim

# install uv / uvx
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# pre-install all mcp servers you want to use and are not part of your source code
RUN uv tool install duckduckgo-mcp-server

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ensure uv is available
ENV PATH="/root/.local/bin:${PATH}"

CMD ["python", "app/mcp_gateway.py"]
