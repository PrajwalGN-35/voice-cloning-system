import os, sys, time, wave, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
OUT = ROOT / "data" / "real_voice_calibration"
OUT.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()

print("=" * 60)
print(" VOICEGUARD — REAL MICROPHONE MODEL DIAGNOSIS")
print("=" * 60)
print()
print("This test DOES NOT change AASIST or its threshold.")
print()
print("Speak normally for each 5-second recording.")
print("You will get 5 recordings.")
print()

# Find a usable microphone through sounddevice.
try:
    import sounddevice as sd
except Exception as e:
    print("sounddevice import failed:", e)
    print()
    print("Installing sounddevice into the existing venv...")
    subprocess.run([str(PY), "-m", "pip", "install", "sounddevice"], check=True)
    import sounddevice as sd

devices = sd.query_devices()
inputs = []

for i, d in enumerate(devices):
    if d.get("max_input_channels", 0) > 0:
        inputs.append((i, d["name"], d["default_samplerate"]))

if not inputs:
    print("MICROPHONE : NOT FOUND")
    sys.exit(2)

print("AVAILABLE MICROPHONES:")
for i, name, rate in inputs:
    print(f"  [{i}] {name} @ {rate:.0f} Hz")

# Prefer a device whose name contains Microphone.
preferred = [x for x in inputs if "microphone" in x[1].lower()]
device = preferred[0] if preferred else inputs[0]

device_id, device_name, device_rate = device

print()
print(f"SELECTED MICROPHONE : {device_name}")
print(f"DEVICE RATE         : {device_rate:.0f} Hz")
print()

# Import existing VoiceGuard detector.
sys.path.insert(0, str(ROOT))
from backend.app.ml.inference.predict import VoiceSpoofingDetector

detector = VoiceSpoofingDetector(
    model_path=str(
        ROOT / "backend" / "app" / "ml" / "checkpoints" / "latest_calibrated.pth"
    )
)

results = []

for n in range(1, 6):
    path = OUT / f"real_voice_{n}.wav"

    print("-" * 60)
    print(f"RECORDING {n}/5")
    print("Speak normally for 5 seconds...")
    time.sleep(1)

    audio = sd.rec(
        int(5 * device_rate),
        samplerate=int(device_rate),
        channels=1,
        dtype="float32",
        device=device_id
    )
    sd.wait()

    audio = np.asarray(audio[:, 0], dtype=np.float32)
    audio = np.nan_to_num(audio)

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio * audio)))

    # Save original microphone capture.
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(device_rate))
        wf.writeframes(pcm.tobytes())

    print(f"Saved : {path.name}")
    print(f"RMS   : {rms:.6f}")
    print(f"Peak  : {peak:.6f}")

    try:
        result = detector.predict(str(path), return_raw=True)

        probs = result["probabilities"]
        real_p = float(probs["real"])
        fake_p = float(probs["fake"])
        pred = result["prediction"]
        conf = float(result["confidence"])
        threshold = float(result["threshold"])

        results.append((n, pred, real_p, fake_p, conf, threshold, rms, peak))

        print(f"PREDICTION : {pred}")
        print(f"CONFIDENCE : {conf:.4f}%")
        print(f"REAL       : {real_p:.8f}")
        print(f"FAKE       : {fake_p:.8f}")
        print(f"THRESHOLD  : {threshold:.8f}")

    except Exception as e:
        print("INFERENCE ERROR:", repr(e))

print()
print("=" * 60)
print(" REAL VOICE DIAGNOSIS")
print("=" * 60)

if not results:
    print("NO INFERENCE RESULTS")
    sys.exit(3)

real_values = np.array([r[2] for r in results])
fake_values = np.array([r[3] for r in results])

print()
print("SAMPLE RESULTS")
print("-" * 60)

for r in results:
    n, pred, real_p, fake_p, conf, threshold, rms, peak = r
    print(
        f"{n}: {pred:4s} | "
        f"REAL={real_p:.6f} | "
        f"FAKE={fake_p:.6f} | "
        f"RMS={rms:.6f}"
    )

print()
print("-" * 60)
print(f"AVERAGE REAL : {real_values.mean():.6f}")
print(f"AVERAGE FAKE : {fake_values.mean():.6f}")
print(f"MIN REAL     : {real_values.min():.6f}")
print(f"MAX REAL     : {real_values.max():.6f}")
print(f"MIN FAKE     : {fake_values.min():.6f}")
print(f"MAX FAKE     : {fake_values.max():.6f}")

fake_count = sum(r[1] == "FAKE" for r in results)
real_count = sum(r[1] == "REAL" for r in results)

print()
print("=" * 60)

if real_count >= 4:
    verdict = "MODEL ACCEPTS YOUR REAL VOICE"
elif fake_count >= 4:
    verdict = "REAL VOICE IS SYSTEMATICALLY CLASSIFIED AS FAKE"
else:
    verdict = "REAL VOICE RESULTS ARE INCONSISTENT"

print("VERDICT :", verdict)
print("=" * 60)

print()
print("IMPORTANT:")
print("These results are from the EXISTING model.")
print("No model weights were changed.")
print("No threshold was changed.")
print()
print("Calibration recordings:")
print(OUT)
print()
