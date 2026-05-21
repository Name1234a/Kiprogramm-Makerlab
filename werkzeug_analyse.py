import json
import re
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


EXPECTED_TOOLS = [
    "hammer",
    "rohrzange",
    "mehrzweckzange",
    "große feile"
]

TOOL_ALIASES = {
    "hammer": ["hammer"],
    "rohrzange": ["wrench", "pipe wrench", "pipe_wrench", "rohr zange"],
    "mehrzweckzange": ["pliers", "multizange", "zange"],
    "grosse feile": ["file", "feile", "große feile", "grosse feile", "grossfeile"]
}


def normalize_name(name: str) -> str:
    normalized = name.lower()
    normalized = normalized.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^a-z0-9 ]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def get_expected_tool_set() -> set[str]:
    expected = {normalize_name(tool) for tool in EXPECTED_TOOLS}
    for key, aliases in TOOL_ALIASES.items():
        expected.add(normalize_name(key))
        for alias in aliases:
            expected.add(normalize_name(alias))
    return expected


def is_expected_tool(tool_name: str) -> bool:
    normalized = normalize_name(tool_name)
    expected_set = get_expected_tool_set()
    return normalized in expected_set


def analyze_tool_recognition() -> dict | None:
    model_path = Path(r"C:\Users\danie\Desktop\KI-Werkzeugwand-Makerlab\Bilder\best.pt")
    image_path = Path(r"C:\Users\danie\Desktop\KI-Werkzeugwand-Makerlab\Bilder\Werk.jpg")
    output_file = Path(r"C:\Users\danie\Desktop\KI-Werkzeugwand-Makerlab\Bilder\werkzeug_analyse.json")
    confidence_threshold = 0.1

    try:
        print(f"Lade Modell aus: {model_path}")
        model = YOLO(str(model_path))
        class_names = model.names if hasattr(model, "names") else {}
        print(f"Modellklassen ({len(class_names)}): {class_names}")

        print(f"Analysiere Bild: {image_path}")
        results = model(str(image_path), imgsz=640, conf=confidence_threshold, verbose=False)

        if not results:
            raise RuntimeError("Keine Ergebnisse vom Modell erhalten.")

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or not hasattr(boxes, "xyxy"):
            raise RuntimeError("Die Ergebnisstruktur enthält keine Box-Daten.")

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        detected_tools = []
        for bbox, conf, cls_id in zip(xyxy, confs, classes):
            tool_name = class_names.get(cls_id, f"class_{cls_id}")
            detected_tools.append({
                "name": tool_name,
                "class_id": int(cls_id),
                "confidence": float(conf),
                "bbox": [float(v) for v in bbox.tolist()]
            })

        print(f"Gefundene Erkennungen: {len(detected_tools)}")
        for tool in detected_tools:
            print(f"  - {tool['name']} (class_id={tool['class_id']}, conf={tool['confidence']:.3f})")

        detected_normalized = {normalize_name(tool["name"]) for tool in detected_tools}

        found_tools = []
        missing_tools = []
        for expected in EXPECTED_TOOLS:
            expected_norm = normalize_name(expected)
            matched = expected_norm in detected_normalized
            if not matched:
                aliases = TOOL_ALIASES.get(expected_norm, [])
                matched = any(normalize_name(alias) in detected_normalized for alias in aliases)
            if matched:
                found_tools.append(expected)
            else:
                missing_tools.append(expected)

        detection_rate = len(found_tools) / len(EXPECTED_TOOLS) if EXPECTED_TOOLS else 0.0

        print("\nAnalyseergebnis:")
        print(f"  Erwartete Werkzeuge: {len(EXPECTED_TOOLS)}")
        print(f"  Gefunden: {len(found_tools)}")
        print(f"  Fehlend: {len(missing_tools)}")
        print(f"  Erkennungsrate: {detection_rate:.1%}")

        if detection_rate >= 0.8:
            rating = "EXZELLENT"
        elif detection_rate >= 0.6:
            rating = "GUT"
        elif detection_rate >= 0.3:
            rating = "BEFRIEDIGEND"
        else:
            rating = "UNGENÜGEND"

        print(f"  Bewertung: {rating}")

        output_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "image_analyzed": str(image_path),
            "total_expected_tools": len(EXPECTED_TOOLS),
            "tools_found": len(found_tools),
            "tools_missing": len(missing_tools),
            "detection_rate": detection_rate,
            "detected_tools": detected_tools,
            "found_tools_list": found_tools,
            "missing_tools_list": missing_tools,
            "model_names": class_names
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Ergebnisse gespeichert in: {output_file}")
        return output_data

    except Exception as e:
        print(f"Fehler bei der Analyse: {e}")
        raise


if __name__ == "__main__":
    analyze_tool_recognition()
