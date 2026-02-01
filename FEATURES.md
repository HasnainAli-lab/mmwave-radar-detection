# Feature Documentation

Complete guide to all 34 features extracted by the mmWave parser.

---

## Feature Categories

### 1. Temporal Features (3)

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `frame` | int | - | Frame sequence number |
| `timestamp` | float | seconds | Time in seconds (CPU cycles / 200MHz) |
| `point_id` | int | - | Point ID within frame |

**Usage:** Time series analysis, tracking, temporal filtering

---

### 2. Spatial Features (7)

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `x` | float | meters | Lateral position (left-right) |
| `y` | float | meters | Longitudinal position (forward-backward) |
| `z` | float | meters | Vertical position (up-down) |
| `range` | float | meters | Euclidean distance from radar |
| `azimuth` | float | degrees | Horizontal angle from boresight (-90° to +90°) |
| `elevation` | float | degrees | Vertical angle |
| `range_idx` | int | - | Raw range bin index |

**Calculation:**
```python
range = sqrt(x² + y² + z²)
azimuth = atan2(x, y) * 180/π
elevation = asin(z / range) * 180/π
```

**Usage:** Object localization, spatial clustering, distance filtering

---

### 3. Velocity Features (5)

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `vx` | float | m/s | Velocity in x direction |
| `vy` | float | m/s | Velocity in y direction |
| `vz` | float | m/s | Velocity in z direction |
| `speed` | float | m/s | Total velocity magnitude |
| `dopplerIdx` | int | - | Doppler bin index |

**Calculation:**
```python
speed = abs(velocity_radial)
vx = speed * (x / range)
vy = speed * (y / range)
vz = speed * (z / range)
```

**Resolution:** 0.005 m/s (corrected from default 0.13 m/s)

**Usage:** Motion detection, velocity filtering, tracking

---

### 4. Acceleration Features (3)

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `ax` | float | m/s² | Acceleration in x direction |
| `ay` | float | m/s² | Acceleration in y direction |
| `az` | float | m/s² | Acceleration in z direction |

**Calculation:**
```python
ax = (vx_current - vx_previous) / dt
ay = (vy_current - vy_previous) / dt
az = (vz_current - vz_previous) / dt
```

**Usage:** Motion classification, sudden movement detection

---

### 5. Signal Quality Features (4)

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `snr` | int | dB | Signal-to-Noise Ratio (from TLV Type 7) |
| `noise` | int | dB | Noise floor level |
| `peakVal` | float | - | Peak signal value (SNR + offset) |
| `RCS` | float | dBsm | Radar Cross Section |

**RCS Calculation:**
```python
RCS = SNR + 40 * log10(range) - 60  # dBsm
```

**Note:** SNR and noise require TLV Type 7 (Side Info) in .dat file. If not present, defaults to 0.

**Usage:** Detection confidence, object size estimation, clutter filtering

---

### 6. Tracking Features (2)

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `trackID` | int | - | Unique track identifier |
| `track_duration` | int | frames | Number of frames track exists |

**Usage:** Multi-object tracking, trajectory analysis

---

### 7. Statistical Features (5)

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `peak_mean` | float | - | Mean peak value across track |
| `peak_std` | float | - | Standard deviation of peak value |
| `speed_mean` | float | m/s | Mean speed across track |
| `speed_std` | float | m/s | Standard deviation of speed |
| `energy_xyz` | float | - | Kinetic energy proxy |

**Energy Calculation:**
```python
energy_xyz = sqrt(vx² + vy² + vz²)
```

**Usage:** Track quality, object characterization

---

### 8. Label Features (5)

| Feature | Type | Description |
|---------|------|-------------|
| `object_type` | string | User-specified object type ('ball', 'lock', etc.) |
| `condition` | string | Auto-detected: 'moving' or 'static' |
| `condition_confidence` | float | Confidence of auto-detection (0-1) |
| `label` | string | Combined label: '{object_type}_{condition}' |
| `azimuth_idx` | int | Raw azimuth bin index |

**Auto-Detection Logic:**
```python
static_percentage = (speed < 0.5).count() / total * 100

if static_percentage > 70:
    condition = 'static'
    confidence = static_percentage / 100
else:
    condition = 'moving'
    confidence = 1 - (static_percentage / 100)
```

**Usage:** Supervised learning labels, filtering

---

## Feature Selection for ML

### Best Features for Different Tasks

#### Moving vs Static Classification
```python
features = ['speed', 'vx', 'vy', 'dopplerIdx', 'energy_xyz', 
            'speed_mean', 'speed_std']
```
**Expected Accuracy:** 95%+

#### Object Type Classification
```python
features = ['range', 'RCS', 'peakVal', 'azimuth', 
            'peak_mean', 'ax', 'ay']
```
**Expected Accuracy:** 70-85%

#### Complete Classification
```python
features = ['range', 'speed', 'vx', 'vy', 'RCS', 'azimuth',
            'ax', 'ay', 'energy_xyz', 'peak_mean']
```
**Expected Accuracy:** 80-90%

---

## Feature Importance Ranking

Based on Random Forest analysis:

1. **speed** - Most discriminative for motion
2. **range** - Spatial filtering
3. **RCS** - Object size/type
4. **vx, vy** - Motion direction
5. **energy_xyz** - Total kinetic energy
6. **azimuth** - Directional info
7. **ax, ay** - Acceleration patterns
8. **speed_mean** - Temporal consistency
9. **peak_mean** - Signal quality
10. **track_duration** - Track reliability

---

## Feature Preprocessing

### Normalization
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Important:** Always normalize features before SVM!

### Missing Values
- `snr`, `noise`: Default to 0 if TLV Type 7 absent
- `ax`, `ay`, `az`: Default to 0 for first frame
- All other features: Should not have missing values

### Outlier Filtering
```python
# Remove points with unrealistic values
df = df[df['range'] < 50]  # Max 50m
df = df[df['speed'] < 20]  # Max 20 m/s
```

---

## Feature Correlation

Highly correlated features (use one from each group):

**Group 1:** speed, energy_xyz  
**Group 2:** vx, vy  
**Group 3:** peak_mean, peakVal  
**Group 4:** speed_mean, speed  

For dimensionality reduction, select one representative from each group.

---

## Adding Custom Features

To add your own features, edit `src/parser.py`:

```python
def _calculate_advanced_features(self, df):
    # ... existing code ...
    
    # Add your custom feature
    df['my_feature'] = df['range'] * df['speed']
    
    return df
```

---

## Feature Versions

| Version | Features | Notes |
|---------|----------|-------|
| 1.0 | 34 | Current version |
| 0.9 | 22 | Beta (missing acceleration) |
| 0.8 | 16 | Initial (basic features only) |

---

## References

- IWR1443 Datasheet: Range resolution = c / (2 * BW)
- Doppler resolution = λ / (2 * T_frame * N_chirps)
- RCS formula from radar equation
