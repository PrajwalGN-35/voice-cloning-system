import sys, wave
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CAL=ROOT/"data"/"real_voice_calibration"
sys.path.insert(0,str(ROOT))
from backend.app.ml.inference.predict import VoiceSpoofingDetector
d=VoiceSpoofingDetector(model_path=str(ROOT/"backend/app/ml/checkpoints/latest_calibrated.pth"))
real_files=sorted(CAL.glob("real_voice_*.wav"))
if len(real_files)<5: raise RuntimeError("Five real_voice_*.wav samples required.")
print("="*65); print(" VOICEGUARD — MODEL SCORE DIAGNOSIS"); print("="*65)
print("\nREAL MICROPHONE SAMPLES")
real=[]
for p in real_files:
 r=d.predict(str(p),return_raw=True); rp=float(r["probabilities"]["real"]); fp=float(r["probabilities"]["fake"]); real.append(fp)
 print(f"{p.name}: REAL={rp:.8f} FAKE={fp:.8f} PRED={r['prediction']}")
print("\nKNOWN TEST FILES")
for p in [ROOT/"test_recording.wav"]:
 if p.exists():
  r=d.predict(str(p),return_raw=True); print(f"{p.name}: REAL={r['probabilities']['real']:.8f} FAKE={r['probabilities']['fake']:.8f} PRED={r['prediction']}")
fake=list(ROOT.rglob("generated_*.wav"))
for p in fake[:5]:
 r=d.predict(str(p),return_raw=True); print(f"{p.name}: REAL={r['probabilities']['real']:.8f} FAKE={r['probabilities']['fake']:.8f} PRED={r['prediction']}")
print("\n" + "="*65)
print(f"MIC REAL-MEAN : {1-np.mean(real):.8f}")
print(f"MIC FAKE-MEAN : {np.mean(real):.8f}")
print(f"MIC FAKE-MIN  : {np.min(real):.8f}")
print(f"MIC FAKE-MAX  : {np.max(real):.8f}")
print(f"CURRENT THRESHOLD : {d.threshold:.8f}")
print("="*65)
print("\nVERDICT:")
if np.mean(real)>0.99:
 print("MODEL/DATASET DOMAIN PROBLEM — REAL VOICE IS CONFIDENTLY REJECTED")
elif np.mean(real)>0.80:
 print("STRONG REAL-VOICE FALSE-POSITIVE PROBLEM")
else:
 print("REAL-VOICE SCORES ARE NOT SYSTEMATICALLY FAKE")
print("\nNO FILES MODIFIED.")
