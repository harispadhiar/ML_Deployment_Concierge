import os
import json
import re
import shutil

SIZE_LIMIT_MB = 10
SIZE_LIMIT_BYTES = SIZE_LIMIT_MB * 1024 * 1024

def detect_framework(model_path: str) -> str:
    """
    Detects the model framework based on the file extension.
    Raises ValueError if file is oversized, doesn't exist, or has unsupported extension.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    file_size = os.path.getsize(model_path)
    if file_size > SIZE_LIMIT_BYTES:
        raise ValueError(f"File size ({file_size / (1024*1024):.2f}MB) exceeds the {SIZE_LIMIT_MB}MB limit.")
        
    _, ext = os.path.splitext(model_path.lower())
    if ext == '.pkl':
        return 'sklearn'
    elif ext in ['.keras', '.h5']:
        return 'keras'
    elif ext in ['.pt', '.pth']:
        return 'pytorch'
    else:
        raise ValueError(f"Unsupported model file extension '{ext}'. Supported extensions: .pkl, .keras, .h5, .pt, .pth")

def generate_requirements(framework: str, extra_packages: list[str] = None) -> list[str]:
    """
    Generates requirements list. Intentionally leaves out joblib for sklearn to demonstrate self-correction!
    """
    reqs = ["gradio", "numpy"]
    if framework == 'sklearn':
        reqs.append("scikit-learn")
        # Intentionally omitting joblib to show self-correction if the app loads via joblib
    elif framework == 'keras':
        reqs.append("tensorflow")
    elif framework == 'pytorch':
        reqs.append("torch")
        
    if extra_packages:
        for pkg in extra_packages:
            if pkg not in reqs:
                reqs.append(pkg)
    return reqs

def generate_gradio_app(framework: str, model_filename: str, inject_audioop: bool = False, inject_bad_import: bool = False) -> str:
    """
    Generates app.py code for Gradio.
    """
    imports = []
    if inject_audioop:
        imports.append("import audioop  # Simulated legacy dependency")
    if inject_bad_import:
        imports.append("import non_existent_package_xyz  # Simulated missing import")
        
    imports_str = "\n".join(imports)
    
    if framework == 'sklearn':
        return f"""import gradio as gr
import pickle
import numpy as np
import joblib  # This will cause ModuleNotFoundError: No module named 'joblib' initially
{imports_str}

# Load the model
try:
    with open("{model_filename}", "rb") as f:
        model = joblib.load(f)
except Exception:
    with open("{model_filename}", "rb") as f:
        model = pickle.load(f)

def predict(input_text):
    try:
        # Expect comma separated numerical values
        data = np.array([float(x.strip()) for x in input_text.split(",")]).reshape(1, -1)
        prediction = model.predict(data)
        return f"Prediction: {{prediction.tolist() if hasattr(prediction, 'tolist') else prediction}}"
    except Exception as e:
        return f"Error: {{str(e)}}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Enter features, e.g., 1.0, 2.0"),
    outputs="text",
    title="ML Deployment Concierge - Scikit-Learn Model"
)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)
"""
    elif framework == 'keras':
        return f"""import gradio as gr
import tensorflow as tf
import numpy as np
{imports_str}

# Load model
model = None
try:
    model = tf.keras.models.load_model("{model_filename}")
except Exception as e:
    # If the file is a mock or corrupted, load_model will raise an error
    pass

def predict(input_text):
    if model is None:
        return "Error: Model not successfully loaded. File may be corrupted or mock."
    try:
        data = np.array([float(x.strip()) for x in input_text.split(",")]).reshape(1, -1)
        prediction = model.predict(data)
        return f"Prediction: {{prediction.tolist()}}"
    except Exception as e:
        return f"Error: {{str(e)}}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Enter features, e.g., 1.0, 2.0"),
    outputs="text",
    title="ML Deployment Concierge - Keras Model"
)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)
"""
    elif framework == 'pytorch':
        return f"""import gradio as gr
import torch
import numpy as np
{imports_str}

# Load model
model = None
try:
    model = torch.load("{model_filename}")
    if hasattr(model, 'eval'):
        model.eval()
except Exception as e:
    pass

def predict(input_text):
    if model is None:
        return "Error: Model not successfully loaded. File may be corrupted or mock."
    try:
        data = np.array([float(x.strip()) for x in input_text.split(",")]).reshape(1, -1)
        tensor_data = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(tensor_data)
        return f"Prediction: {{prediction.tolist()}}"
    except Exception as e:
        return f"Error: {{str(e)}}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Enter features, e.g., 1.0, 2.0"),
    outputs="text",
    title="ML Deployment Concierge - PyTorch Model"
)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)
"""
    else:
        raise ValueError(f"Unknown framework: {framework}")

def generate_dockerfile(framework: str, python_version: str = "3.11-slim") -> str:
    return f"""FROM python:{python_version}

WORKDIR /code

# Install system dependencies if any
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt --break-system-packages

COPY . .

CMD ["python", "app.py"]
"""

def generate_readme(framework: str, model_filename: str) -> str:
    return f"""# ML Deployment Bundle

This bundle was automatically generated by the **ML Deployment Concierge**.

## Model Information
- **Framework:** {framework}
- **Model File:** {model_filename}

## Local Execution
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Gradio app:
   ```bash
   python app.py
   ```

## Docker Deployment
```bash
docker build -t ml-model-app .
docker run -p 7860:7860 ml-model-app
```
"""

def write_files(bundle_dir: str, app_code: str, requirements: list[str], dockerfile_code: str, readme_code: str):
    os.makedirs(bundle_dir, exist_ok=True)
    
    with open(os.path.join(bundle_dir, "app.py"), "w", encoding="utf-8") as f:
        f.write(app_code)
        
    with open(os.path.join(bundle_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(requirements) + "\n")
        
    with open(os.path.join(bundle_dir, "Dockerfile"), "w", encoding="utf-8") as f:
        f.write(dockerfile_code)
        
    with open(os.path.join(bundle_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_code)

def builder_revise(bundle_dir: str, framework: str, error_report: dict, skill_memory_path: str) -> tuple[list[str], str, str]:
    """
    Revises requirements, dockerfile, or app.py based on the error report and memory.
    """
    # Load memory
    memory = {"fixes": []}
    if os.path.exists(skill_memory_path):
        try:
            with open(skill_memory_path, "r", encoding="utf-8") as f:
                memory = json.load(f)
        except Exception:
            pass
            
    # Load current files
    req_path = os.path.join(bundle_dir, "requirements.txt")
    docker_path = os.path.join(bundle_dir, "Dockerfile")
    app_path = os.path.join(bundle_dir, "app.py")
    
    current_reqs = []
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            current_reqs = [line.strip() for line in f if line.strip()]
            
    current_docker = ""
    if os.path.exists(docker_path):
        with open(docker_path, "r", encoding="utf-8") as f:
            current_docker = f.read()
            
    current_app = ""
    if os.path.exists(app_path):
        with open(app_path, "r", encoding="utf-8") as f:
            current_app = f.read()

    error_msg = error_report.get("error_message", "")
    error_cat = error_report.get("category", "")
    
    applied_fix = False
    
    # 1. Search skill memory for matches
    for fix in memory.get("fixes", []):
        pattern = fix.get("error_pattern", "")
        if pattern.lower() in error_msg.lower():
            # Apply memory fix
            action = fix.get("action", {})
            
            # Update requirements
            for req in action.get("requirements", []):
                # Clean versioned package comparison
                pkg_name = req.split("==")[0].split(">=")[0].strip()
                # Remove old reference if present
                current_reqs = [r for r in current_reqs if not r.startswith(pkg_name)]
                current_reqs.append(req)
                
            # Update dockerfile
            df_line = action.get("dockerfile", "")
            if df_line and "FROM" in df_line:
                # Replace the FROM line
                lines = current_docker.splitlines()
                for i, line in enumerate(lines):
                    if line.startswith("FROM"):
                        lines[i] = df_line
                        break
                current_docker = "\n".join(lines) + "\n"
                
            # If the fix requires modifying app code (e.g. audioop stub or removal)
            if "audioop" in pattern.lower():
                # For local running, we can remove 'import audioop' from app.py or stub it
                current_app = current_app.replace("import audioop", "# import audioop (removed due to Python 3.13+ incompatibility)")
                
            applied_fix = True
            break
            
    # 2. Heuristics fallback if memory didn't match
    if not applied_fix:
        if error_cat == "missing_import" or "ModuleNotFoundError" in error_msg:
            # Try to extract the module name
            # Format usually: "ModuleNotFoundError: No module named 'joblib'"
            import re
            match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_msg)
            if match:
                missing_pkg = match.group(1)
                # Map typical import names to pip package names if needed
                pkg_map = {
                    "sklearn": "scikit-learn",
                    "tf": "tensorflow",
                    "joblib": "joblib",
                    "audioop": "audioop-lts" # there is a backport for Python 3.13+ called audioop-lts!
                }
                pip_pkg = pkg_map.get(missing_pkg, missing_pkg)
                if pip_pkg not in current_reqs:
                    current_reqs.append(pip_pkg)
                    applied_fix = True
                    
        elif "audioop" in error_msg:
            # Specific local fallback for audioop
            current_app = current_app.replace("import audioop", "# import audioop (removed)")
            applied_fix = True

    return current_reqs, current_docker, current_app


def generate_hf_readme(framework: str, model_filename: str, space_name: str) -> str:
    """
    Generates a HuggingFace Spaces README with required YAML frontmatter.
    HF Spaces needs this to detect the SDK and set the app file.
    """
    return f"""---
title: {space_name}
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# {space_name}

Auto-generated by **ML Deployment Concierge**.

- **Framework:** {framework.upper()}
- **Model:** `{model_filename}`

Upload feature values (comma-separated numbers) and click Submit to get a prediction.
"""


def deploy_to_hf_spaces(
    bundle_dir: str,
    hf_token: str,
    space_id: str,
    framework: str,
    model_filename: str,
    private: bool = False,
) -> dict:
    """
    Pushes the validated deployment bundle to a HuggingFace Space.

    Args:
        bundle_dir:      Local path to the validated bundle directory.
        hf_token:        HuggingFace write-access token (hf_...).
        space_id:        Full space id, e.g. "username/my-model-space".
        framework:       Detected framework string (sklearn / keras / pytorch).
        model_filename:  Model file name (already copied inside bundle_dir).
        private:         If True the Space will be private.

    Returns:
        dict with keys: success (bool), url (str), error (str).
    """
    try:
        from huggingface_hub import HfApi, SpaceHardware
    except ImportError:
        return {"success": False, "url": "", "error": "huggingface_hub is not installed. Run: pip install huggingface_hub"}

    if not hf_token or not hf_token.strip().startswith("hf_"):
        return {"success": False, "url": "", "error": "Invalid HF token — must start with 'hf_'. Generate one at https://huggingface.co/settings/tokens"}

    parts = space_id.strip().split("/")
    if len(parts) != 2 or not all(parts):
        return {"success": False, "url": "", "error": f"space_id must be 'username/space-name', got: '{space_id}'"}

    api = HfApi(token=hf_token)

    # 1. Overwrite README.md with HF-compatible frontmatter
    hf_readme_path = os.path.join(bundle_dir, "README.md")
    with open(hf_readme_path, "w", encoding="utf-8") as f:
        f.write(generate_hf_readme(framework, model_filename, parts[1]))

    # 2. Create (or get) the Space repo
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            private=private,
            exist_ok=True,   # idempotent — safe to call on existing Space
        )
    except Exception as e:
        return {"success": False, "url": "", "error": f"Failed to create Space repo '{space_id}': {e}"}

    # 3. Upload the entire bundle directory
    try:
        api.upload_folder(
            folder_path=bundle_dir,
            repo_id=space_id,
            repo_type="space",
            commit_message="Deployed by ML Deployment Concierge",
            ignore_patterns=["*.venv*", ".venv_test", "__pycache__", "*.pyc", "temp_requirements.txt"],
        )
    except Exception as e:
        return {"success": False, "url": "", "error": f"Failed to upload bundle: {e}"}

    space_url = f"https://huggingface.co/spaces/{space_id}"
    return {"success": True, "url": space_url, "error": ""}
