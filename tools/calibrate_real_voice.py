import sys, os, wave, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CAL=ROOT/"data"/"real_voice_calibration"
FAKE=CAL/"synthetic_fake"
FAKE.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))

from backend.app.ml.inference.predict import VoiceSpoofingDetector

print("="*70)
print(" VOICEGUARD — CONTROLLED REAL/Fake SCORE CALIBRATION")
print("="*70)

detector=VoiceSpoofingDetector(
    model_path=str(ROOT/"backend"/"app"/"ml"/"checkpoints"/"latest_calibrated.pth")
)

real_files=sorted(CAL.glob("real_voice_*.wav"))
if len(real_files)<5:
    raise RuntimeError("Expected 5 real microphone recordings.")

# Create five controlled synthetic non-speech signals.
sr=16000
duration=5
t=np.arange(sr*duration,dtype=np.float32)/sr

fake_files=[]
for i in range(5):
    rng=np.random.default_rng(100+i)
    # Mixture of tones + noise + modulation to avoid a trivial single-tone sample.
    x=(
        0.12*np.sin(2*np.pi*(180+35*i)*t) +
        0.07*np.sin(2*np.pi*(420+27*i)*t) +
        0.025*rng.normal(size=len(t)).astype(np.float32)
    )
    x=np.clip(x,-0.95,0.95)
    pcm=(x*32767).astype(np.int16)
    p=FAKE/f"synthetic_fake_{i+1}.wav"
    with wave.open(str(p),"wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    fake_files.append(p)

def score(path):
    r=detector.predict(str(path),return_raw=True)
    return float(r["probabilities"]["fake"]),float(r["probabilities"]["real"])

real=[]
fake=[]

print("\nREAL MICROPHONE SCORES")
print("-"*70)
for p in real_files:
    f,r=score(p)
    real.append(f)
    print(f"{p.name:22s} REAL={r:.8f}  FAKE={f:.8f}")

print("\nCONTROLLED FAKE SCORES")
print("-"*70)
for p in fake_files:
    f,r=score(p)
    fake.append(f)
    print(f"{p.name:22s} REAL={r:.8f}  FAKE={f:.8f}")

real=np.array(real)
fake=np.array(fake)

print("\nSCORE DISTRIBUTION")
print("-"*70)
print(f"REAL mean : {real.mean():.8f}")
print(f"REAL min  : {real.min():.8f}")
print(f"REAL max  : {real.max():.8f}")
print(f"FAKE mean : {fake.mean():.8f}")
print(f"FAKE min  : {fake.min():.8f}")
print(f"FAKE max  : {fake.max():.8f}")

# Candidate threshold maximising separation on the collected calibration set.
all_scores=np.concatenate([real,fake])
candidates=np.unique(np.concatenate([
    np.linspace(max(0,float(all_scores.min())-0.01),1.0,1001),
    all_scores
]))

best=None
for th in candidates:
    real_errors=np.sum(real>=th)
    fake_errors=np.sum(fake<th)
    errors=int(real_errors+fake_errors)
    if best is None or errors<best[0]:
        best=(errors,float(th),int(real_errors),int(fake_errors))

errors,th,real_errors,fake_errors=best
accuracy=(len(real)+len(fake)-errors)/(len(real)+len(fake))

print("\nCALIBRATION")
print("-"*70)
print(f"OLD MODEL THRESHOLD : {detector.threshold:.8f}")
print(f"CALIBRATED THRESHOLD: {th:.8f}")
print(f"CALIBRATION ACCURACY : {accuracy*100:.2f}%")
print(f"REAL MISCLASSIFIED   : {real_errors}/{len(real)}")
print(f"FAKE MISCLASSIFIED   : {fake_errors}/{len(fake)}")

# Safety: refuse to modify production logic if calibration cannot separate the
# collected classes. This prevents a misleading "fix".
if real.max() >= fake.min():
    print("\nVERDICT : NO SAFE THRESHOLD SEPARATES THESE SAMPLES")
    print("PRODUCTION CODE : NOT MODIFIED")
    print("Reason: genuine microphone scores overlap controlled-fake scores.")
    print("A threshold change would trade false REAL/Fake decisions.")
    sys.exit(10)

# Write calibration configuration only when there is actual separation.
cfg=ROOT/"backend"/"app"/"ml"/"inference"/"calibration_config.py"
cfg.write_text(
f'''# Auto-generated from controlled VoiceGuard calibration.\\n'
f'# Created: {__import__("datetime").datetime.now().isoformat()}\\n'
f'CALIBRATED_FAKE_THRESHOLD = {th!r}\\n'
f'CALIBRATION_REAL_MAX = {float(real.max())!r}\\n'
f'CALIBRATION_FAKE_MIN = {float(fake.min())!r}\\n'
'''.replace("'\\n","\\n"),
encoding="utf-8"
)

predict=ROOT/"backend"/"app"/"ml"/"inference"/"predict.py"
text=predict.read_text(encoding="utf-8")

if "calibration_config" not in text:
    marker="from pathlib import Path"
    inject=(
        "from pathlib import Path\n"
        "try:\n"
        "    from backend.app.ml.inference.calibration_config import CALIBRATED_FAKE_THRESHOLD\n"
        "except Exception:\n"
        "    CALIBRATED_FAKE_THRESHOLD = None\n"
    )
    text=text.replace(marker,inject,1)

# Only replace the decision comparison, preserving the model and probabilities.
old="if probabilities[\"fake\"] >= self.threshold:"
new="if probabilities[\"fake\"] >= (CALIBRATED_FAKE_THRESHOLD if CALIBRATED_FAKE_THRESHOLD is not None else self.threshold):"

if old not in text:
    print("DECISION PATCH TARGET NOT FOUND")
    print("PRODUCTION CODE : NOT MODIFIED")
    sys.exit(11)

text=text.replace(old,new,1)
predict.write_text(text,encoding="utf-8")

print("\nPRODUCTION CALIBRATION : APPLIED")
print(f"Decision threshold : {th:.8f}")
print("AASIST WEIGHTS      : UNCHANGED")
print("MODEL ARCHITECTURE  : UNCHANGED")
print("="*70)
