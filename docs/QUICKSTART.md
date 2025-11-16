# Quick Start Guide

## Installation (5 minutes)

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd Multimodal-Emotion-Aware-AI-Companion
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env if needed (defaults work fine)
```

### 3. Run
```bash
python app.py
```

Visit `http://localhost:8501` in your browser.

## First Steps

1. **Try the chat** - Type a message with emotion
   - "I'm feeling really stressed today"
   - "This is amazing, I love it!"

2. **View your mood** - Check the emotion gauge on the right

3. **Explore dashboard** - Click the "Dashboard" tab to see analytics

4. **Run demo** - Test individual components:
   ```bash
   python demo.py
   ```

## Configuration Tips

### Use a smaller model (faster)
Edit `.env`:
```
LLM_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Use GPU acceleration
Edit `.env`:
```
LLM_DEVICE=cuda
```

### Adjust emotion detection
Edit `config/config.yaml`:
```yaml
emotion_detection:
  face:
    confidence_threshold: 0.5  # Lower = more sensitive
  text:
    confidence_threshold: 0.3
```

## Troubleshooting

### Models downloading slowly?
The first run downloads models (~2-4GB). Be patient.

### Out of memory?
Use a smaller LLM model (see above).

### Webcam not working?
Disable facial detection in sidebar for now.

### PyAudio installation fails?
Voice emotion is optional. Comment out `pyaudio` in `requirements.txt`.

## Next Steps

- Read the full [README.md](../README.md)
- Customize `config/config.yaml`
- Check out the [Architecture Guide](ARCHITECTURE.md)
- Explore the code in `src/`

## Docker Alternative

If you prefer Docker:

```bash
docker-compose up
```

That's it! Visit `http://localhost:8501`
