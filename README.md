# ZED Foxglove Integration

A multi-camera streaming application that captures data from Stereolabs ZED cameras and streams it to Foxglove for real-time visualization and recording. The project includes both Python and C++ implementations for maximum flexibility.

## Prerequisites

## Installation

### Python Version

1. **Install ZED SDK** following the [official installation guide](https://www.stereolabs.com/docs/installation/)

2. **Navigate to the Python directory**:
   ```bash
   cd python/
   ```

3. **Install Python dependencies** using Poetry:
   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install
   ```

   Or using pip:
   ```bash
   pip install foxglove-sdk>=0.4.0 numpy>=2.0 opencv-python>=4.10 requests>=2.31
   ```

4. **Install ZED Python API**: The project includes `pyzed-5.0-cp310-cp310-linux_x86_64.whl`. Install it with:
   ```bash
   pip install pyzed-5.0-cp310-cp310-linux_x86_64.whl
   ```

### C++ Version

1. **Install ZED SDK** as above

2. **Navigate to the C++ directory**:
   ```bash
   cd cpp/
   ```

3. **Build the project**:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

## Usage

### Python Version

```bash
cd python/

# Stream to Foxglove Studio via WebSocket
poetry run python main.py --ws

# Record to MCAP file
poetry run python main.py --mcap output.mcap

# Both streaming and recording
poetry run python main.py --ws --mcap output.mcap
```

### C++ Version

```bash
cd cpp/build/

# Stream to Foxglove Studio via WebSocket
./zed_foxglove --ws

# Record to MCAP file
./zed_foxglove --mcap output.mcap

# Both streaming and recording
./zed_foxglove --ws --mcap output.mcap
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ws` | Enable WebSocket server for live streaming | Disabled |
| `--mcap <file>` | Output MCAP file name | `output.mcap` (C++), disabled (Python) |
| `--help`, `-h` | Show help message | - |

## Foxglove Studio Configuration

1. **Start Foxglove Studio**

2. **Connect to live stream**:
   - Click "Open connection"
   - Select "Foxglove WebSocket"
   - Use URL: `ws://localhost:8765`

3. **Open recorded data**:
   - Click "Open local file"
   - Select your `.mcap` file

4. **Add panels** to visualize:
   - **Image Panel**: Subscribe to `/image_0`, `/image_1`, etc. for camera images
   - **Image Panel**: Subscribe to `/depth_0`, `/depth_1`, etc. for depth maps
   - **3D Panel**: Subscribe to `/point_cloud_0`, `/point_cloud_1`, etc. for point clouds
