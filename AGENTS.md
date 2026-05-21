# AGENTS.md

## Project Overview
This is a YOLOv8-based tool detection system for identifying tools on a MakerLab wall. The main script `werkzeug_analyse.py` analyzes images and generates detection reports.

## Key Conventions
- Use German for user-facing messages and tool names.
- Apply type hints for clarity.
- Use dictionary-based configurations for multilingual support.
- Log every major step with `print()` for debugging.

## Common Pitfalls
- **Hardcoded paths**: Model and image paths are absolute and machine-specific. Always check `C:\Users\danie\Downloads\` references.
- **Missing dependencies**: Manually install `ultralytics`, `torch`, `torchvision`, `pillow`.
- **Model path mismatch**: Code expects model in Downloads, but `Bilder/best.pt` exists locally.
- **Low detection rate**: Current ~30% suggests model/data mismatch; compare `EXPECTED_TOOLS` with `model.names`.

## Debugging Agent
When handling bug information from console output:
1. Parse error messages for specific patterns:
   - "Keine Ergebnisse vom Modell" → Model loading failed.
   - "Die Ergebnisstruktur enthält keine Box-Daten" → YOLO output format issue.
   - Low "Erkennungsrate" → Accuracy problem.
2. Check hardcoded paths first.
3. Validate tool class names match aliases after normalization.
4. Suggest moving paths to config file or using relative paths.
5. Recommend adding requirements.txt and proper logging.

For more details, see [README.md](README.md).</content>
<parameter name="filePath">c:\Users\danie\Desktop\Kiprogramm-Makerlab\AGENTS.md