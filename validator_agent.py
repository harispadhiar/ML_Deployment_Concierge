import os
import sys
import subprocess
import time
import shutil
import re
import ast
import threading
import queue

def get_venv_paths(workspace_dir: str) -> tuple[str, str, str]:
    """
    Returns the path to the virtual environment, python executable, and pip executable.
    """
    venv_dir = os.path.join(workspace_dir, ".venv_test")
    if sys.platform == "win32":
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        python_exe = os.path.join(venv_dir, "bin", "python")
        pip_exe = os.path.join(venv_dir, "bin", "pip")
    return venv_dir, python_exe, pip_exe

def create_stubs_if_needed(venv_dir: str, requirements: list[str]):
    """
    Creates lightweight stubs for heavy ML packages (tensorflow, torch) 
    if they are required but not installed. This prevents multi-gigabyte downloads 
    during local validation while still verifying imports and launch logic.
    """
    # Find site-packages directory in the venv
    site_packages = None
    if sys.platform == "win32":
        sp_dir = os.path.join(venv_dir, "Lib", "site-packages")
        if os.path.exists(sp_dir):
            site_packages = sp_dir
    else:
        lib_dir = os.path.join(venv_dir, "lib")
        if os.path.exists(lib_dir):
            # Find python3.x folder
            for sub in os.listdir(lib_dir):
                if sub.startswith("python"):
                    sp_dir = os.path.join(lib_dir, sub, "site-packages")
                    if os.path.exists(sp_dir):
                        site_packages = sp_dir
                        break
                        
    if not site_packages:
        return
        
    for req in requirements:
        req_lower = req.lower()
        if "tensorflow" in req_lower:
            tf_dir = os.path.join(site_packages, "tensorflow")
            if not os.path.exists(tf_dir):
                os.makedirs(tf_dir, exist_ok=True)
                # Create __init__.py with mock structures
                with open(os.path.join(tf_dir, "__init__.py"), "w", encoding="utf-8") as f:
                    f.write("""# Mock TensorFlow
class MockKerasModels:
    def load_model(self, *args, **kwargs):
        raise Exception("Mock Keras model loader")
class MockKeras:
    models = MockKerasModels()
keras = MockKeras()
""")
        elif "torch" in req_lower and not "torchvision" in req_lower:
            torch_dir = os.path.join(site_packages, "torch")
            if not os.path.exists(torch_dir):
                os.makedirs(torch_dir, exist_ok=True)
                with open(os.path.join(torch_dir, "__init__.py"), "w", encoding="utf-8") as f:
                    f.write("""# Mock PyTorch
def load(*args, **kwargs):
    raise Exception("Mock PyTorch model loader")
class tensor:
    def __init__(self, *args, **kwargs):
        pass
class no_grad:
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_val, exc_tb): pass
float32 = "float32"
""")
        elif "torchvision" in req_lower:
            tv_dir = os.path.join(site_packages, "torchvision")
            if not os.path.exists(tv_dir):
                os.makedirs(tv_dir, exist_ok=True)
                with open(os.path.join(tv_dir, "__init__.py"), "w", encoding="utf-8") as f:
                    f.write("# Mock Torchvision\n")

def run_validation(bundle_dir: str, workspace_dir: str, timeout: int = 15) -> tuple[bool, str]:
    """
    Sets up a virtual environment, installs requirements, runs app.py in a subprocess,
    and smoke tests if the Gradio application starts up successfully.
    """
    venv_dir, python_exe, pip_exe = get_venv_paths(workspace_dir)
    
    # 1. Ensure venv exists — inherit system site-packages so gradio/numpy are already present
    if not os.path.exists(python_exe):
        print(f"Creating virtual environment in {venv_dir}...")
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", venv_dir],
            check=True
        )
        
    # 2. Read requirements from bundle
    req_file = os.path.join(bundle_dir, "requirements.txt")
    if not os.path.exists(req_file):
        return False, "requirements.txt not found in bundle."
        
    with open(req_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip()]
        
    # 3. Create stubs for heavy ML libraries to keep the test rapid
    create_stubs_if_needed(venv_dir, requirements)
    
    # 4. Install requirements in venv
    print("Installing requirements...")
    # On Windows, pip install might fail with break-system-packages warning or similar, but in venv it's fine.
    # We install the requirements line by line or all at once.
    # Let's filter out heavy packages from pip install since we stubbed them!
    install_reqs = []
    for req in requirements:
        req_name = req.split("==")[0].split(">=")[0].strip().lower()
        if req_name not in ["tensorflow", "torch", "torchvision"]:
            install_reqs.append(req)
            
    if install_reqs:
        # Write a temporary requirements file without tf/torch to install
        temp_req = os.path.join(bundle_dir, "temp_requirements.txt")
        with open(temp_req, "w", encoding="utf-8") as f:
            f.write("\n".join(install_reqs) + "\n")
        try:
            res = subprocess.run([pip_exe, "install", "-r", temp_req], capture_output=True, text=True)
            if res.returncode != 0:
                return False, f"pip install failed:\nStdout: {res.stdout}\nStderr: {res.stderr}"
        finally:
            if os.path.exists(temp_req):
                os.remove(temp_req)
                
    # 5. Launch the Gradio app in the subprocess
    app_script = os.path.join(bundle_dir, "app.py")
    if not os.path.exists(app_script):
        return False, "app.py not found in bundle."
        
    print(f"Starting Gradio app smoke test: {app_script}...")
    
    # We copy the model file to the bundle directory if it's not already there
    # The app.py will load it from the bundle directory
    
    # Run the app.py using python_exe
    # We set environment variables to avoid Gradio analytics/updates and make it headless
    env = os.environ.copy()
    env["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    proc = subprocess.Popen(
        [python_exe, "-u", "app.py"],
        cwd=bundle_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env
    )
    
    # Monitor output for successful startup or crash
    start_time = time.time()
    success = False
    logs = []
    
    q = queue.Queue()
    def read_output(pipe):
        for line in iter(pipe.readline, ''):
            q.put(line)
        pipe.close()
        
    t = threading.Thread(target=read_output, args=(proc.stdout,))
    t.daemon = True
    t.start()
    
    try:
        while time.time() - start_time < timeout:
            ret_code = proc.poll()
            if ret_code is not None:
                # Process crashed or exited
                break
                
            try:
                line = q.get(timeout=0.5)
                if line:
                    logs.append(f"STDOUT: {line.strip()}")
                    if "running on local url" in line.lower() or "http://" in line.lower():
                        success = True
                        break
            except queue.Empty:
                pass
                
    except Exception as e:
        logs.append(f"Validator Exception: {str(e)}")
    finally:
        # Clean up process
        if proc.poll() is None:
            print("Terminating Gradio app subprocess...")
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                
        # Capture remaining logs
        remaining_out, remaining_err = proc.communicate()
        if remaining_out:
            logs.append(remaining_out)
        if remaining_err:
            logs.append(remaining_err)
            
    full_logs = "\n".join(logs)
    return success, full_logs

def classify_error(logs: str) -> dict:
    """
    Analyzes the validation logs to classify the error type.
    """
    report = {
        "category": "unknown",
        "error_message": "",
        "suggested_fix": ""
    }
    
    # Search for Python tracebacks or error messages
    tb_match = re.findall(r"(?:Traceback \(most recent call last\):.*?\n)(.*?)(?=\n\w|\Z)", logs, re.DOTALL)
    if tb_match:
        report["error_message"] = tb_match[-1].strip()
    else:
        # Fallback to last few lines of the logs
        lines = [line.strip() for line in logs.splitlines() if line.strip()]
        report["error_message"] = "\n".join(lines[-5:]) if lines else "No error logs captured."
        
    error_msg = report["error_message"]
    
    if "ModuleNotFoundError" in error_msg or "ImportError" in error_msg:
        if "audioop" in error_msg:
            report["category"] = "missing_audioop"
            report["suggested_fix"] = "Pin python to 3.11-slim or remove audioop usage"
        elif "imp" in error_msg:
            report["category"] = "missing_imp"
            report["suggested_fix"] = "Pin python to 3.11-slim"
        else:
            report["category"] = "missing_import"
            match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_msg)
            missing = match.group(1) if match else "unknown_module"
            report["suggested_fix"] = f"Add {missing} to requirements.txt"
            
    elif "VersionConflict" in error_msg or "ResolutionPossible" in error_msg:
        report["category"] = "dependency_conflict"
        report["suggested_fix"] = "Resolve dependency version pin"
        
    elif "UnpicklingError" in error_msg or "BadZipFile" in error_msg or "OSError: SavedModel file does not exist" in error_msg or "corrupted" in error_msg.lower():
        report["category"] = "corrupted_model"
        report["suggested_fix"] = "Reject model file as corrupted or invalid structure"
        
    return report


def fast_check(bundle_dir: str) -> tuple[bool, str]:
    """
    Fast static validation — no subprocess or venv required.
    1. Parses app.py with ast.parse (catches SyntaxErrors).
    2. Confirms requirements.txt is non-empty.
    3. Walks the AST for non-heavy imports and checks availability.
    Returns (success, log_string).
    """
    logs = []

    # Check app.py syntax
    app_path = os.path.join(bundle_dir, "app.py")
    if not os.path.exists(app_path):
        return False, "app.py not found in bundle."
    with open(app_path, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        ast.parse(source)
        logs.append("app.py: syntax OK")
    except SyntaxError as e:
        return False, f"SyntaxError in app.py: {e}"

    # Detect bad imports by scanning the AST
    tree = ast.parse(source)
    bad_imports = []
    heavy = {"tensorflow", "torch", "torchvision", "sklearn", "joblib", "tf"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                top = (name or "").split(".")[0]
                if top and top not in heavy:
                    try:
                        __import__(top)
                    except ImportError:
                        bad_imports.append(top)

    if bad_imports:
        msg = f"ModuleNotFoundError: No module named '{bad_imports[0]}'"
        logs.append(f"Missing imports detected: {bad_imports}")
        return False, "\n".join(logs) + "\n" + msg

    # Check requirements.txt
    req_path = os.path.join(bundle_dir, "requirements.txt")
    if not os.path.exists(req_path):
        return False, "requirements.txt not found in bundle."
    with open(req_path, "r", encoding="utf-8") as f:
        reqs = [l.strip() for l in f if l.strip()]
    if not reqs:
        return False, "requirements.txt is empty."
    logs.append(f"requirements.txt: {len(reqs)} packages listed")

    return True, "\n".join(logs)
