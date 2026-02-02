# Client-Server Mapping Architecture

This document explains the client-server architecture for advanced mapping capabilities, allowing you to offload computationally intensive mapping to a separate server.

## Overview

The client-server architecture separates:
- **Client** (runs on robot): Captures RGBD frames, controls robot, visualization
- **Server** (runs anywhere): Performs advanced volumetric mapping (e.g., wavemap)

This allows you to:
- Run heavy computation on a powerful workstation/server
- Keep the robot-side code lightweight and responsive
- Easily swap different mapping backends without changing robot code
- Scale to multiple robots mapping to one server

## Architecture

```
Robot (Client)                          Server (Wavemap)
┌─────────────────┐                     ┌──────────────────┐
│ Camera Loop     │                     │                  │
│   ↓             │                     │  Volumetric      │
│ CameraState ────┼──→ ZeroMQ REQ  ───→ │  Mapping         │
│                 │    (RGB, Depth,     │  (wavemap)       │
│ Odometry Loop   │     T_world_cam)    │                  │
│   ↓             │                     │                  │
│ OdometryState ←─┼──← ZeroMQ REP  ←─── │  Point Clouds    │
│   ↓             │    (occupied,       │  (occupied,      │
│ Visualization   │     free voxels)    │   free)          │
└─────────────────┘                     └──────────────────┘
```

## Communication Protocol

### Technology: ZeroMQ

We use **ZeroMQ** for communication because:
- **Fast**: Near-zero overhead, optimized for high-throughput
- **Reliable**: Built-in reconnection and error handling
- **Simple**: Clean request-reply pattern
- **Flexible**: Works over TCP (localhost or network)
- **Battle-tested**: Used widely in robotics and real-time systems

### Message Format

Messages use **msgpack** with **msgpack-numpy** for efficient serialization:
- Compact binary format (smaller than JSON)
- Native numpy array support (zero-copy when possible)
- Optional zlib compression for images

**Request (Client → Server):**
```python
{
    "rgb": np.ndarray,           # RGB image (H, W, 3) uint8
    "depth": np.ndarray,         # Depth image (H, W) float32
    "T_world_cam": np.ndarray,   # Camera pose (4, 4) float64
    "timestamp": float,          # Frame timestamp
    "rgb_shape": tuple,          # Image dimensions
    "depth_shape": tuple         # Depth dimensions
}
```

**Response (Server → Client):**
```python
{
    "success": bool,             # Whether mapping succeeded
    "occupied_points": np.ndarray,   # Occupied voxels (N, 3) float32
    "free_points": np.ndarray,       # Free voxels (M, 3) float32
    "occupied_colors": np.ndarray,   # Colors for occupied (N, 3) uint8
    "error_msg": str             # Error message if failed
}
```

## Dependencies

Install the required packages:

```bash
pip install pyzmq msgpack msgpack-numpy
```

## Usage

### 1. Start the Wavemap Server

In one terminal:

```bash
python -m reachy2_stack.base_module.wavemap_server --port 5555 --voxel-size 0.05
```

Options:
- `--port`: ZeroMQ port (default: 5555)
- `--voxel-size`: Voxel size in meters (default: 0.05)

### 2. Run the Client

In another terminal:

```bash
python reachy2_stack/tests/module_test_client_server.py
```

The client will:
- Connect to robot and start camera/odometry loops
- Send RGBD frames to wavemap server at configured rate
- Receive occupied/free voxel point clouds
- Visualize in Open3D

## Configuration

Edit [`module_test_client_server.py`](../reachy2_stack/tests/module_test_client_server.py):

```python
# Server connection
MAPPING_SERVER_HOST = "localhost"  # Change to server IP if remote
MAPPING_SERVER_PORT = 5555
MAPPING_HZ = 2.0                   # Request rate (lower = less bandwidth)
MAPPING_TIMEOUT_MS = 5000          # Connection timeout
```

## Integrating Your Wavemap

The server includes a placeholder for wavemap integration. Replace the placeholder in [`wavemap_server.py`](../reachy2_stack/base_module/wavemap_server.py):

```python
def __init__(self, port: int = 5555, voxel_size: float = 0.05):
    # TODO: Replace with your actual wavemap initialization
    from your_module import WaveMapVolumetricMapper
    self.mapper = WaveMapVolumetricMapper(voxel_size=voxel_size)

def process_frame(self, rgb, depth, T_world_cam, timestamp):
    # Update map with new observation
    self.mapper.integrate_frame(rgb, depth, T_world_cam)

    # Extract voxels
    occupied_points, occupied_colors = self.mapper.get_occupied_voxels()
    free_points = self.mapper.get_free_voxels()

    return occupied_points, free_points, occupied_colors
```

## Performance Considerations

### Bandwidth Optimization

1. **Compression**: Messages are compressed with zlib (level 1 = fast)
2. **Rate limiting**: Adjust `MAPPING_HZ` to balance freshness vs bandwidth
3. **Image downsampling**: Optionally downsample images before sending

### Latency

- **Localhost**: ~1-5ms round-trip
- **LAN**: ~5-20ms round-trip (depends on network)
- **Timeout**: Set `MAPPING_TIMEOUT_MS` based on expected latency + processing time

### Error Handling

The client automatically:
- Reconnects after timeouts
- Resets socket on errors
- Stops after 5 consecutive errors

## Network Setup

### Same Machine (Localhost)
```python
MAPPING_SERVER_HOST = "localhost"
```

### Different Machines (LAN)
```python
# On server machine, note its IP address (e.g., 192.168.1.100)
# On client machine:
MAPPING_SERVER_HOST = "192.168.1.100"
```

Ensure firewall allows TCP traffic on the chosen port.

## Advanced: Multiple Clients

The current REQ/REP pattern supports one client. For multiple clients, use:

1. **ROUTER/DEALER pattern** for connection pooling
2. **PUB/SUB pattern** for broadcasting map updates
3. **Separate request queues** per client

See ZeroMQ documentation for these patterns.

## Debugging

### Check Server is Running
```bash
# Should show process listening on port
netstat -an | grep 5555
```

### Test Connection
```python
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://localhost:5555")
sock.send(b"ping")
print(sock.recv())  # Should receive response
```

### Common Issues

1. **Timeout errors**: Increase `MAPPING_TIMEOUT_MS` or reduce `MAPPING_HZ`
2. **Connection refused**: Check server is running and firewall settings
3. **Slow performance**: Check network bandwidth, consider downsampling images
4. **Memory issues**: Server may need to limit map size or implement pruning

## Comparison with Local Mapping

| Aspect | Local (`mapping_loop`) | Client-Server (`mapping_loop_client`) |
|--------|----------------------|---------------------------------------|
| **Latency** | ~0ms | ~5-20ms |
| **CPU Load** | On robot | On server |
| **Complexity** | Simple | Moderate |
| **Flexibility** | Limited | High (swap backends) |
| **Scalability** | Single robot | Multiple robots |
| **Best for** | Simple mapping | Advanced/heavy mapping |

## Future Enhancements

1. **Bidirectional updates**: Server can push map updates without request
2. **Map persistence**: Save/load volumetric maps
3. **Multi-resolution**: Adaptive voxel sizes based on distance
4. **Semantic integration**: Combine with object detection
5. **Loop closure**: Server detects and handles loop closures
