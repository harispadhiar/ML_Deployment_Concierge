import os
import json
import time
import shutil
from datetime import datetime
import builder_agent as builder
import validator_agent as validator

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def log_step(logs_list: list, agent: str, tool: str, input_data: any, output_data: any):
    step = {
        "agent": agent,
        "tool_called": tool,
        "input": input_data,
        "output": output_data,
        "timestamp": datetime.now().isoformat()
    }
    logs_list.append(step)
    return step

def update_skill_memory(memory_path: str, error_class: str, error_pattern: str, fix_desc: str, requirements: list, dockerfile: str):
    """
    Appends a new successful fix to skill_memory.json.
    """
    memory = {"fixes": []}
    if os.path.exists(memory_path):
        try:
            with open(memory_path, "r", encoding="utf-8") as f:
                memory = json.load(f)
        except Exception:
            pass
            
    # Check if this error_class is already in memory
    for fix in memory["fixes"]:
        if fix.get("error_class") == error_class:
            # Already exists, update it
            fix["action"] = {
                "requirements": requirements,
                "dockerfile": dockerfile
            }
            break
    else:
        # Add new fix
        new_fix = {
            "error_class": error_class,
            "error_pattern": error_pattern,
            "fix_description": fix_desc,
            "action": {
                "requirements": requirements,
                "dockerfile": dockerfile
            }
        }
        memory["fixes"].append(new_fix)
        
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

def run_orchestrator(
    model_path: str,
    workspace_dir: str,
    max_retries: int = 3,
    hf_token: str = "",
    space_id: str = "",
):
    """
    Generator that runs the multi-agent deployment concierge pipeline.
    Yields dict steps for UI/event updates.
    At the end, writes structured logs to logs/run_<timestamp>.json.
    """
    run_logs = []
    run_id = get_timestamp()
    os.makedirs(os.path.join(workspace_dir, "logs"), exist_ok=True)
    log_file_path = os.path.join(workspace_dir, "logs", f"run_{run_id}.json")
    
    yield {"type": "status", "agent": "Orchestrator", "message": f"Initializing run {run_id}...", "retry": 0}
    
    # 1. Guardrail checks
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
            
        file_size = os.path.getsize(model_path)
        if file_size > builder.SIZE_LIMIT_BYTES:
            raise ValueError(f"File size ({file_size / (1024*1024):.2f}MB) exceeds the {builder.SIZE_LIMIT_MB}MB limit.")
            
        framework = builder.detect_framework(model_path)
        yield {"type": "status", "agent": "Orchestrator", "message": f"Guardrails passed. Framework detected: {framework.upper()}", "retry": 0}
        log_step(run_logs, "Orchestrator", "guardrails_and_framework_detection", {"model_path": model_path}, {"framework": framework, "file_size": file_size})
    except Exception as e:
        err_msg = str(e)
        yield {"type": "error", "agent": "Orchestrator", "message": f"Guardrail/Detection failed: {err_msg}"}
        log_step(run_logs, "Orchestrator", "guardrails_and_framework_detection", {"model_path": model_path}, {"error": err_msg})
        # Write log file even on immediate failure
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(run_logs, f, indent=2)
        return

    # 2. Create bundle directory
    bundle_dir = os.path.join(workspace_dir, f"bundle_{run_id}")
    os.makedirs(bundle_dir, exist_ok=True)
    
    # Copy model file to bundle directory
    model_filename = os.path.basename(model_path)
    dest_model_path = os.path.join(bundle_dir, model_filename)
    shutil.copy2(model_path, dest_model_path)
    
    # Determine if we should inject test failures
    inject_audioop = "conflict" in model_filename.lower()
    inject_bad_import = "corrupted" in model_filename.lower()
    
    # 3. Builder generates initial bundle
    yield {"type": "status", "agent": "Builder", "message": "Generating initial deployment bundle...", "retry": 0}
    
    requirements = builder.generate_requirements(framework)
    app_code = builder.generate_gradio_app(framework, model_filename, inject_audioop=inject_audioop, inject_bad_import=inject_bad_import)
    dockerfile = builder.generate_dockerfile(framework)
    readme = builder.generate_readme(framework, model_filename)
    
    builder.write_files(bundle_dir, app_code, requirements, dockerfile, readme)
    
    log_step(run_logs, "Builder", "write_initial_files", 
             {"framework": framework, "model_filename": model_filename, "inject_audioop": inject_audioop}, 
             {"requirements": requirements, "dockerfile": dockerfile})
    
    # 4. Validation & Self-Correction Loop
    retries = 0
    success = False
    last_error_class = None
    last_error_message = None
    last_fix_applied = None
    
    skill_memory_path = os.path.join(workspace_dir, "skill_memory.json")
    
    while retries <= max_retries:
        yield {"type": "status", "agent": "Validator", "message": f"Running smoke-test validation (Attempt {retries + 1}/{max_retries + 1})...", "retry": retries}
        
        # Run validation instantly using AST check
        val_success, val_logs = validator.fast_check(bundle_dir)
        log_step(run_logs, "Validator", "fast_check", {"bundle_dir": bundle_dir, "attempt": retries}, {"success": val_success, "logs": val_logs})
        
        if val_success:
            yield {"type": "status", "agent": "Validator", "message": "Validation succeeded! Smoke-test passed successfully.", "retry": retries}
            success = True
            
            # If we fixed an error, save it to skill memory
            if retries > 0 and last_error_class:
                yield {"type": "status", "agent": "Builder", "message": f"Learning new pattern. Saving fix for '{last_error_class}' to skill memory...", "retry": retries}
                # Read final requirements and dockerfile
                with open(os.path.join(bundle_dir, "requirements.txt"), "r", encoding="utf-8") as f:
                    final_reqs = [line.strip() for line in f if line.strip()]
                with open(os.path.join(bundle_dir, "Dockerfile"), "r", encoding="utf-8") as f:
                    final_df = f.read()
                    
                update_skill_memory(
                    skill_memory_path,
                    error_class=last_error_class,
                    error_pattern=last_error_message,
                    fix_desc=last_fix_applied,
                    requirements=final_reqs,
                    dockerfile=final_df
                )
                log_step(run_logs, "Orchestrator", "update_skill_memory", 
                         {"error_class": last_error_class, "error_pattern": last_error_message}, 
                         {"status": "memory_updated"})
            break
            
        # Validation failed, check if we can retry
        retries += 1
        if retries > max_retries:
            yield {"type": "status", "agent": "Orchestrator", "message": "Max retries reached. Validation failed.", "retry": retries - 1}
            break
            
        # Classify the error
        error_report = validator.classify_error(val_logs)
        last_error_class = error_report["category"]
        last_error_message = error_report["error_message"]
        last_fix_applied = error_report["suggested_fix"]
        
        yield {
            "type": "status", 
            "agent": "Validator", 
            "message": f"Validation failed! Category: {last_error_class.upper()}. Msg: {last_error_message.splitlines()[0] if last_error_message else 'Unknown'}", 
            "retry": retries - 1
        }
        log_step(run_logs, "Validator", "classify_error", {"logs": val_logs}, {"report": error_report})
        
        # If it is a corrupted model, stop immediately because self-correction cannot fix corrupted file bytes
        if last_error_class == "corrupted_model":
            yield {"type": "error", "agent": "Orchestrator", "message": "Validation failed immediately: Model file is corrupted or has an invalid internal structure. Stopping pipeline."}
            log_step(run_logs, "Orchestrator", "terminate_corrupted_model", {}, {"message": "Corrupted model detected, stopping."})
            break
            
        # Builder revises files
        yield {"type": "status", "agent": "Builder", "message": f"Analyzing failure and applying fix (suggested: {last_fix_applied})...", "retry": retries}
        
        rev_reqs, rev_df, rev_app = builder.builder_revise(bundle_dir, framework, error_report, skill_memory_path)
        builder.write_files(bundle_dir, rev_app, rev_reqs, rev_df, readme)
        
        log_step(run_logs, "Builder", "builder_revise", 
                 {"error_report": error_report, "bundle_dir": bundle_dir}, 
                 {"requirements": rev_reqs, "dockerfile": rev_df})
        
    # Write execution log
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(run_logs, f, indent=2)
        
    if success:
        # Zip the bundle
        zip_path = os.path.join(workspace_dir, f"bundle_{run_id}.zip")
        shutil.make_archive(os.path.join(workspace_dir, f"bundle_{run_id}"), 'zip', bundle_dir)

        # Optional: deploy to HF Spaces
        space_url = ""
        if hf_token and space_id:
            yield {"type": "status", "agent": "Orchestrator",
                   "message": f"Deploying bundle to HuggingFace Space '{space_id}'...", "retry": retries}
            deploy_result = builder.deploy_to_hf_spaces(
                bundle_dir=bundle_dir,
                hf_token=hf_token,
                space_id=space_id,
                framework=framework,
                model_filename=model_filename,
            )
            log_step(run_logs, "Orchestrator", "deploy_to_hf_spaces",
                     {"space_id": space_id},
                     {"success": deploy_result["success"], "url": deploy_result["url"],
                      "error": deploy_result["error"]})
            if deploy_result["success"]:
                space_url = deploy_result["url"]
                yield {"type": "status", "agent": "Orchestrator",
                       "message": f"Space live at: {space_url}", "retry": retries}
            else:
                yield {"type": "status", "agent": "Orchestrator",
                       "message": f"HF deploy warning: {deploy_result['error']} (bundle zip still available)",
                       "retry": retries}

        yield {
            "type": "success",
            "agent": "Orchestrator",
            "message": f"Pipeline finished! Bundle zipped as bundle_{run_id}.zip"
                       + (f" | Space: {space_url}" if space_url else ""),
            "zip_path": zip_path,
            "bundle_dir": bundle_dir,
            "log_file": log_file_path,
            "space_url": space_url,
        }
    else:
        yield {
            "type": "error",
            "agent": "Orchestrator",
            "message": f"Pipeline execution failed. Review the logs at logs/run_{run_id}.json for details.",
            "log_file": log_file_path
        }
