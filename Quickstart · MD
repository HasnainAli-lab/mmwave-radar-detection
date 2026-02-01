# Quick Start Guide

Get started with mmWave radar data processing in 5 minutes!

## Installation

```bash
git clone https://github.com/yourusername/mmwave-radar-detection.git
cd mmwave-radar-detection
pip install -r requirements.txt
```

## Step 1: Prepare Your Data

Place your `.dat` files in the `data/raw/` directory:

```
data/raw/
├── ball1.dat
├── ball2.dat
├── lock1.dat
└── lock2.dat
```

## Step 2: Parse Single File

```python
from src.parser import PerfectParser

parser = PerfectParser()
df = parser.parse('data/raw/ball1.dat', object_type='ball')

print(f"Extracted {len(df)} points")
print(f"Condition: {df['condition'].iloc[0]}")
```

**Output:**
```
Extracted 240 points
Condition: moving
```

## Step 3: Batch Process All Files

```python
from src.batch_processor import BatchProcessor

processor = BatchProcessor()
processor.process_directory('data/raw/', object_types={
    'ball*.dat': 'ball',
    'lock*.dat': 'lock'
})

df_all = processor.get_combined_dataset()
```

## Step 4: Train ML Model

```python
from src.ml_trainer import SVMTrainer

trainer = SVMTrainer()
trainer.load_data('data/processed/combined_dataset.csv')

# Train classifier
accuracy = trainer.train_moving_vs_static()
print(f"Accuracy: {accuracy*100:.2f}%")

# Save model
trainer.save_model('moving_vs_static')
```

## Step 5: Use Trained Model

```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model
with open('data/models/moving_vs_static_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/models/moving_vs_static_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict on new data
new_data = pd.read_csv('data/processed/new_file.csv')
X = new_data[['range', 'speed', 'vx', 'vy', 'RCS', 'azimuth', 'energy_xyz', 'doppler_idx']]
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)

print(predictions)
```

## Common Tasks

### View All Features
```python
df = pd.read_csv('data/processed/ball_complete.csv')
print(df.columns.tolist())
```

### Filter by Condition
```python
moving = df[df['condition'] == 'moving']
static = df[df['condition'] == 'static']
```

### Calculate Statistics
```python
print(df.groupby('condition')['speed'].mean())
```

### Export for Excel
```python
df.to_excel('radar_data.xlsx', index=False)
```

## Next Steps

- Read [FEATURES.md](docs/FEATURES.md) for detailed feature documentation
- Check [examples/](examples/) for more usage examples
- See [notebooks/](notebooks/) for Jupyter notebook tutorials

## Troubleshooting

**Problem:** No points extracted  
**Solution:** Check TLV type compatibility. Your file might use a different format.

**Problem:** Speed values too high  
**Solution:** Velocity resolution is set to 0.005 m/s. Adjust if needed in `src/parser.py`.

**Problem:** SNR = 0  
**Solution:** Your .dat file doesn't contain TLV Type 7. This is normal for some configurations.

## Support

- GitHub Issues: [Create an issue](https://github.com/yourusername/mmwave-radar-detection/issues)
- Email: your.email@example.com
