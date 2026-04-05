# Mango Leaf Disease Detection (Defense Demo Package)

This workspace is prepared for second defense demo on April 6, 2026.

## What is included

- `mango_disease_detection.py`: End-to-end pipeline
  - VGG16 + Logistic Regression + PSO design
  - Works in demo mode without dataset (synthetic hardcoded feature data)
  - Switches to real-image mode automatically if dataset + dependencies are present
- `requirements.txt`: Minimal dependencies for demo mode
- `defense_materials.md`: 12-slide content, contribution split, checklist, and Q&A

## Quick start (Windows, VS Code terminal)

1. Create environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Run demo mode directly:

   ```powershell
   python mango_disease_detection.py --demo
   ```

## Output files

After run completes, these files are generated:

- `mango_disease_model.pkl`
- `feature_scaler.pkl`
- `run_metadata.pkl`

## Real dataset mode (optional)

Create folder structure:

```text
mango_leaf_dataset/
├── Healthy/
├── Anthracnose/
├── Powdery_Mildew/
├── Sooty_Mold/
├── Bacterial_Canker/
├── Rust/
├── Leaf_Blight/
└── Dieback/
```

Install optional packages and run:

```powershell
pip install opencv-python tensorflow
python mango_disease_detection.py --data-dir mango_leaf_dataset
```
