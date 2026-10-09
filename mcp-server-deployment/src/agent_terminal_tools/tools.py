"""Common terminal / shell tools for agents, exposed as a FastMCP server.

This module defines the FastMCP server instance ``mcp`` and the actual tools.
Each tool is a small, single-purpose function decorated with ``@mcp.tool()``.
It takes a few simple arguments, performs the operation, and returns a single
human-readable string — the command / file output on success, or a clear
message describing the error on failure. Returning a string (instead of
raising) lets an agent always read what happened and react in the same turn.

``agent_terminal_tools/main.py`` imports ``mcp`` from here and runs it.
"""

import glob
import os
import shlex
import subprocess
import sys

from fastmcp import FastMCP

mcp = FastMCP(
    name="agent-terminal-tools",
    instructions=(
        "Tools for running common shell commands and file operations: "
        "execute a bash command, run files and Python scripts, list "
        "directories, glob and grep files, and read, write, append, delete, "
        "copy, move, create and inspect files and folders. Every tool returns "
        "a single string containing either the result or a clear error "
        "message."
    ),
)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _run_command(command, use_shell: bool = False) -> str:
    """Run a command and return one decoded string describing the outcome.

    ``command`` is either a single string (``use_shell=True``) or a list of
    argv tokens (``use_shell=False``). Both stdout and stderr are captured so a
    failing command can always be diagnosed. On success this returns the
    command's stdout (with a trailing stderr block included if any was written);
    on failure it returns the exit code together with stderr.
    """
    label = command if use_shell else " ".join(map(str, command))
    try:
        result = subprocess.run(
            command,
            shell=use_shell,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as err:
        stderr = (err.stderr or b"").decode("utf-8", errors="replace").strip()
        body = f"Command failed with exit code {err.returncode}: {label}"
        if stderr:
            body += f"\n{stderr}"
        return body
    except (FileNotFoundError, PermissionError) as err:
        return f"Could not run the command ({label}): {err}"

    stdout = (result.stdout or b"").decode("utf-8", errors="replace").strip()
    stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
    if stdout and stderr:
        return f"{stdout}\n[stderr]\n{stderr}"
    if stdout:
        return stdout
    if stderr:
        return stderr
    return "Command completed successfully (no output)."


# ---------------------------------------------------------------------------
# Shell execution
# ---------------------------------------------------------------------------

@mcp.tool()
def bash(command: str):
    '''This function will execute a bash command and return the output. This function is useful for executing bash commands including creating files and directories'''
    return _run_command(command, use_shell=True)


@mcp.tool()
def execute_file(file_path: str, args: str = ""):
    '''Use bash to execute a file or script and return the output. Provide any arguments for the file as a space-separated string in args (e.g. args="foo bar").'''
    return _run_command(f"bash {shlex.quote(file_path)} {args}".strip(), use_shell=True)


@mcp.tool()
def run_python(script: str, args: str = ""):
    '''Execute a Python script using the same interpreter that runs this server and return the output. Provide script arguments as a space-separated string in args (e.g. args="a=b --flag").'''
    return _run_command(f"{shlex.quote(sys.executable)} {shlex.quote(script)} {args}".strip(), use_shell=True)


# ---------------------------------------------------------------------------
# Discovering files
# ---------------------------------------------------------------------------

@mcp.tool()
def list_directory(path: str = "."):
    '''List the files and subdirectories inside a directory (ls -la). Returns the listing, including hidden files with permissions, size, and modification time.'''
    if not os.path.isdir(path):
        return f"Not a directory: {path}"
    return _run_command(["ls", "-la", "--color=never", path])


@mcp.tool()
def glob_files(pattern: str, path: str = ".", recursive: bool = True):
    '''Find files by glob pattern (e.g. **/*.py or src/*.js) and return the matching file paths, one per line. Set recursive=False to only match a single directory level.'''
    target = pattern if os.path.isabs(pattern) else os.path.join(os.path.abspath(path), pattern)
    matches = sorted(glob.glob(target, recursive=recursive))
    if not matches:
        return f"No files matched the pattern: {pattern}"
    return "\n".join(matches)


@mcp.tool()
def grep(pattern: str, path: str = ".", include: str = ""):
    '''Search file contents for a regex pattern (like: grep -rn) and return the matching lines with file name and line number. Optionally restrict the search to files whose names match an include glob (e.g. include="*.py").'''
    command = ["grep", "-rn"]
    if include:
        command.append(f"--include={include}")
    command += [pattern, path]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except (FileNotFoundError, PermissionError) as err:
        return f"Could not run grep: {err}"

    text = (result.stdout or b"").decode("utf-8", errors="replace").strip()
    if result.returncode == 0:
        return text or f"No matches found for /{pattern}/ in {path}"
    if result.returncode == 1:
        return f"No matches found for /{pattern}/ in {path}"

    stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
    return f"grep failed (exit code {result.returncode}): {stderr}"


# ---------------------------------------------------------------------------
# File contents
# ---------------------------------------------------------------------------

@mcp.tool()
def read_file(file_path: str):
    '''Read a text file and return its full contents as a string. Use this to view the current contents of a file before editing it.'''
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except IsADirectoryError:
        return f"Is a directory, not a file: {file_path}. Use list_directory instead."
    except OSError as err:
        return f"Could not read {file_path}: {err}"


@mcp.tool()
def write_file(file_path: str, content: str):
    '''Write contents to a file as text, creating any missing parent directories and overwriting any existing file. Returns a confirmation with how many characters were written.'''
    try:
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as err:
        return f"Could not write to {file_path}: {err}"
    return f"Wrote {len(content)} characters to {file_path}."


@mcp.tool()
def append_file(file_path: str, content: str):
    '''Append text to the end of a file, creating the file (and any missing parent directories) if it does not exist yet. Returns a confirmation with how many characters were appended.'''
    try:
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as err:
        return f"Could not append to {file_path}: {err}"
    return f"Appended {len(content)} characters to {file_path}."


# ---------------------------------------------------------------------------
# Files and folders
# ---------------------------------------------------------------------------

@mcp.tool()
def delete_file(file_path: str):
    '''Delete (unlink) a file and return a confirmation (rm -v). Refuses to delete a directory; use delete_folder for those. This is destructive and cannot be undone.'''
    if os.path.isdir(file_path):
        return f"Refusing to delete a directory with delete_file. Use delete_folder instead. Path: {file_path}"
    return _run_command(["rm", "-v", file_path])


@mcp.tool()
def create_folder(folder_path: str):
    '''Create a directory (folder), including any missing parent directories (mkdir -p). Returns confirmation of the created path, or a note if it already existed.'''
    if os.path.exists(folder_path) and not os.path.isdir(folder_path):
        return f"Cannot create folder: {folder_path} already exists and is not a directory."
    out = _run_command(["mkdir", "-v", "-p", folder_path])
    if out.startswith("Command completed successfully"):
        return f"Folder ready: {os.path.abspath(folder_path)}"
    return out


@mcp.tool()
def delete_folder(folder_path: str):
    '''Recursively delete a directory (folder) and everything inside it (rm -r). This is destructive and cannot be undone. Returns a confirmation.'''
    if not os.path.isdir(folder_path):
        return f"Not a directory: {folder_path}"
    out = _run_command(["rm", "-r", folder_path])
    if out.startswith("Command completed successfully"):
        return f"Deleted folder and contents: {os.path.abspath(folder_path)}"
    return out


@mcp.tool()
def copy_file(source: str, destination: str):
    '''Copy a file or directory to a new location (cp -r). Returns a confirmation or the error message.'''
    return _run_command(["cp", "-r", source, destination])


@mcp.tool()
def move_file(source: str, destination: str):
    '''Rename or move a file or directory to a new location (mv). Returns a confirmation or the error message.'''
    return _run_command(["mv", source, destination])


@mcp.tool()
def file_info(file_path: str):
    '''Show detailed metadata for a file or directory (stat): size, type, permissions, owner, and modification time. Useful for checking existence without opening the file.'''
    return _run_command(["stat", file_path])
