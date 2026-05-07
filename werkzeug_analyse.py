import torch
import json
import sys
import os
from collections import defaultdict

def analyze_tool_recognition():
    """
    Analysiert die Ausgaben des YOLOv5-Modells und bewertet die Werkzeug-Erkennung
    """

    # Pfade
    model_path = r"C:\Users\danie\Downloads\best.pt"
    image_path = r"C:\Users\danie\Downloads\Werk.jpg"

    # Erwartete Werkzeuge (passe diese Liste an deine Bedürfnisse an)
    EXPECTED_TOOLS = [
        "hammer", "wrench", "screwdriver", "pliers", "drill",
        "saw", "tape_measure", "level", "multimeter",
        "soldering_iron", "file", "brush", "sandpaper"
    ]

    try:
        # Modell-Checkpoint laden
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

        # Klassennamen aus den Trainingsargumenten extrahieren
        class_names = []
        if 'train_args' in ckpt:
            train_args = ckpt['train_args']
            if hasattr(train_args, 'names') and train_args.names:
                class_names = train_args.names
            elif hasattr(train_args, 'data') and train_args.data:
                # Versuche aus data-Pfad zu extrahieren
                try:
                    import yaml
                    with open(train_args.data, 'r') as f:
                        data = yaml.safe_load(f)
                    if 'names' in data:
                        class_names = data['names']
                except:
                    pass

        if not class_names and 'names' in ckpt:
            class_names = ckpt['names']

        if not class_names:
            # Fallback: Verwende Standard-Werkzeug-Klassen
            print("Warnung: Klassennamen konnten nicht extrahiert werden - verwende Standard-Werkzeug-Klassen")
            class_names = EXPECTED_TOOLS + ["unbekannt"] * (80 - len(EXPECTED_TOOLS))

        print(f"Erkannte Klassen: {len(class_names)}")
        print(f"Klassennamen: {class_names[:10]}...")  # Erste 10 anzeigen

        # Modell laden
        model = ckpt['model']
        model.eval()

        # Bild laden und vorverarbeiten
        from PIL import Image
        from torchvision import transforms

        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

        input_tensor = transform(image).unsqueeze(0).half()

        # Erkennung durchführen
        with torch.no_grad():
            results = model(input_tensor)

        # Erkennungen analysieren - verbesserte Verarbeitung
        detected_tools = []
        confidence_threshold = 0.1  # Sehr niedrige Schwelle zum Testen

        print(f"Verarbeite Ergebnisse vom Typ: {type(results)}")

        # Debug: Zeige Struktur der Ergebnisse
        if isinstance(results, tuple):
            print(f"Tuple mit {len(results)} Elementen")
            for i, item in enumerate(results):
                print(f"  Element {i}: Typ {type(item)}, Shape: {getattr(item, 'shape', 'N/A')}")

        # YOLOv5-Ausgabe verarbeiten
        if isinstance(results, tuple) and len(results) >= 1:
            predictions = results[0]  # Erste Element ist normalerweise die Predictions

            if hasattr(predictions, 'shape') and len(predictions.shape) == 3:
                print(f"Predictions Shape: {predictions.shape}")

                # Ultra-Layers: [batch, channels, anchors]
                # Wir permuten, um [batch, anchors, channels] zu erhalten.
                predictions = predictions.permute(0, 2, 1)
                print(f"Permutierte Predictions Shape: {predictions.shape}")

                batch_size, num_anchors, features = predictions.shape

                for batch_idx in range(min(batch_size, 1)):
                    for anchor_idx in range(num_anchors):
                        detection = predictions[batch_idx, anchor_idx]

                        if len(detection) >= 5:
                            x, y, w, h, obj = detection[:5]
                            class_scores = detection[5:]

                            # Wenn keine Class-Scores im Tensor sind, versuche alternative Scores
                            if len(class_scores) == 0 and isinstance(results[1], dict) and 'scores' in results[1]:
                                scores = results[1]['scores'][0]
                                if anchor_idx < scores.shape[1]:
                                    class_scores = scores[:, anchor_idx]

                            if len(class_scores) == 0:
                                continue

                            best_class_idx = int(class_scores.argmax())
                            best_class_score = float(class_scores[best_class_idx])
                            combined_conf = float(obj) * best_class_score

                            if combined_conf >= confidence_threshold:
                                if best_class_idx < len(class_names):
                                    tool_name = class_names[best_class_idx]
                                else:
                                    tool_name = f"class_{best_class_idx}"

                                x1 = float(x - w / 2)
                                y1 = float(y - h / 2)
                                x2 = float(x + w / 2)
                                y2 = float(y + h / 2)

                                detected_tools.append({
                                    'name': tool_name,
                                    'confidence': combined_conf,
                                    'bbox': [x1, y1, x2, y2],
                                    'raw_conf': float(obj),
                                    'class_conf': best_class_score,
                                    'class_id': best_class_idx
                                })

        # Alternative: Wenn results ein Dictionary ist
        elif isinstance(results, dict):
            print("Dictionary-Format erkannt")
            for key, value in results.items():
                print(f"  {key}: {type(value)}, Shape: {getattr(value, 'shape', 'N/A')}")

        print(f"Verarbeitete {len(detected_tools)} potenzielle Erkennungen")

        # Debug: Zeige alle Erkennungen
        if detected_tools:
            print("Gefundene Erkennungen:")
            for tool in detected_tools[:5]:  # Erste 5 anzeigen
                print(f"  {tool['name']}: {tool['confidence']:.3f} (raw: {tool.get('raw_conf', 'N/A'):.3f})")

        # Ergebnisse analysieren und bewerten
        print("\n" + "="*50)
        print("WERKZEUG-ERKENNUNGS-ANALYSE")
        print("="*50)

        # Erkannte Werkzeuge anzeigen
        print(f"\nErkannte Werkzeuge ({len(detected_tools)}):")
        for tool in detected_tools:
            print(f"  ✓ {tool['name']} (Konfidenz: {tool['confidence']:.1%})")

        # Vergleich mit erwarteten Werkzeugen
        detected_names = [tool['name'].lower() for tool in detected_tools]
        expected_lower = [tool.lower() for tool in EXPECTED_TOOLS]

        found_tools = []
        missing_tools = []

        for expected in EXPECTED_TOOLS:
            if expected.lower() in detected_names:
                found_tools.append(expected)
            else:
                missing_tools.append(expected)

        print(f"\nVon {len(EXPECTED_TOOLS)} erwarteten Werkzeugen gefunden:")
        print(f"  ✓ Vorhanden: {len(found_tools)}")
        for tool in found_tools:
            print(f"    - {tool}")

        print(f"  ✗ Fehlend: {len(missing_tools)}")
        for tool in missing_tools:
            print(f"    - {tool}")

        # Bewertung berechnen
        detection_rate = len(found_tools) / len(EXPECTED_TOOLS) if EXPECTED_TOOLS else 0

        print("\nBewertung:")
        if detection_rate >= 0.8:
            print(f"  🟢 EXZELLENT: {detection_rate:.1%} der erwarteten Werkzeuge erkannt")
        elif detection_rate >= 0.6:
            print(f"  🟡 GUT: {detection_rate:.1%} der erwarteten Werkzeuge erkannt")
        elif detection_rate >= 0.3:
            print(f"  🟠 BEFRIEDIGEND: {detection_rate:.1%} der erwarteten Werkzeuge erkannt")
        else:
            print(f"  🔴 UNGENÜGEND: Nur {detection_rate:.1%} der erwarteten Werkzeuge erkannt")

        # Zusätzliche Statistiken
        if detected_tools:
            avg_confidence = sum(tool['confidence'] for tool in detected_tools) / len(detected_tools)
            print(f"Durchschnittliche Konfidenz: {avg_confidence:.1f}")
            max_confidence = max(tool['confidence'] for tool in detected_tools)
            min_confidence = min(tool['confidence'] for tool in detected_tools)
            print(f"Konfidenz-Bereich: {min_confidence:.1f} - {max_confidence:.1f}")
        # JSON-Ausgabe für weitere Verarbeitung
        output_data = {
            'analysis_timestamp': '2026-05-07',
            'image_analyzed': image_path,
            'total_expected_tools': len(EXPECTED_TOOLS),
            'tools_found': len(found_tools),
            'tools_missing': len(missing_tools),
            'detection_rate': detection_rate,
            'detected_tools': detected_tools,
            'found_tools_list': found_tools,
            'missing_tools_list': missing_tools
        }

        # Speichere Ergebnisse als JSON
        output_file = r"C:\Users\danie\Downloads\werkzeug_analyse.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nErgebnisse gespeichert in: {output_file}")

        return output_data

    except Exception as e:
        print(f"Fehler bei der Analyse: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyze_tool_recognition()