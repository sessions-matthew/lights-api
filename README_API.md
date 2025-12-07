# Triones Controller REST API

A REST API backend for controlling Triones RGBW Bluetooth LED controllers. This API does not maintain persistent device connections - each request connects to the device, executes the command, and disconnects.

## Installation

```bash
# Install dependencies
pip install -r api-requirements.txt

# Also ensure triones dependencies are installed
pip install -r requirements.txt
```

## Running the API

```bash
# Development mode
python api.py

# Or with uvicorn directly
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Discovery

**GET /devices**
Discover Triones devices on the network

Query Parameters:
- `timeout` (float, optional): Discovery timeout in seconds (1-30, default: 10)

Response:
```json
[
  {
    "name": "Triones-XXXX",
    "address": "AA:BB:CC:DD:EE:FF"
  }
]
```

### Device Status

**GET /devices/{address}/status**
Get current device status

Response:
```json
{
  "is_on": true,
  "mode": 37,
  "speed": 1,
  "red": 255,
  "green": 0,
  "blue": 0,
  "white": 0,
  "rgb_hex": "#FF0000"
}
```

### Power Control

**POST /devices/{address}/power/on**
Turn device on

**POST /devices/{address}/power/off**
Turn device off

Response:
```json
{
  "success": true,
  "message": "Device powered on"
}
```

### Color Control

**POST /devices/{address}/color/rgb**
Set RGB color

Request Body:
```json
{
  "red": 255,
  "green": 0,
  "blue": 0
}
```

**POST /devices/{address}/color/hex**
Set color using hex string

Request Body:
```json
{
  "color": "#FF0000"
}
```

**POST /devices/{address}/color/white**
Set white mode

Request Body:
```json
{
  "intensity": 255
}
```

### Built-in Modes

**POST /devices/{address}/mode**
Set built-in lighting mode

Request Body:
```json
{
  "mode": 37,
  "speed": 1
}
```

**GET /modes**
List all available built-in modes

Response:
```json
{
  "SEVEN_COLOR_CROSS_FADE": 37,
  "RED_GRADUAL_CHANGE": 38,
  ...
}
```

## Usage Examples

### Using curl

```bash
# Discover devices
curl http://localhost:8000/devices

# Get device status
curl http://localhost:8000/devices/AA:BB:CC:DD:EE:FF/status

# Turn on
curl -X POST http://localhost:8000/devices/AA:BB:CC:DD:EE:FF/power/on

# Set RGB color
curl -X POST http://localhost:8000/devices/AA:BB:CC:DD:EE:FF/color/rgb \
  -H "Content-Type: application/json" \
  -d '{"red": 255, "green": 0, "blue": 0}'

# Set hex color
curl -X POST http://localhost:8000/devices/AA:BB:CC:DD:EE:FF/color/hex \
  -H "Content-Type: application/json" \
  -d '{"color": "#00FF00"}'

# Set built-in mode
curl -X POST http://localhost:8000/devices/AA:BB:CC:DD:EE:FF/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": 37, "speed": 10}'

# Turn off
curl -X POST http://localhost:8000/devices/AA:BB:CC:DD:EE:FF/power/off
```

### Using Python requests

```python
import requests

BASE_URL = "http://localhost:8000"

# Discover devices
devices = requests.get(f"{BASE_URL}/devices").json()
address = devices[0]["address"]

# Turn on and set color
requests.post(f"{BASE_URL}/devices/{address}/power/on")
requests.post(
    f"{BASE_URL}/devices/{address}/color/rgb",
    json={"red": 255, "green": 128, "blue": 0}
)

# Get status
status = requests.get(f"{BASE_URL}/devices/{address}/status").json()
print(f"Current color: {status['rgb_hex']}")
```

## Available Modes

Mode values range from 37 to 56:

- 37: SEVEN_COLOR_CROSS_FADE
- 38: RED_GRADUAL_CHANGE
- 39: GREEN_GRADUAL_CHANGE
- 40: BLUE_GRADUAL_CHANGE
- 41: YELLOW_GRADUAL_CHANGE
- 42: CYAN_GRADUAL_CHANGE
- 43: PURPLE_GRADUAL_CHANGE
- 44: WHITE_GRADUAL_CHANGE
- 45: RED_GREEN_CROSS_FADE
- 46: RED_BLUE_CROSS_FADE
- 47: GREEN_BLUE_CROSS_FADE
- 48: SEVEN_COLOR_STROBE_FLASH
- 49: RED_STROBE_FLASH
- 50: GREEN_STROBE_FLASH
- 51: BLUE_STROBE_FLASH
- 52: YELLOW_STROBE_FLASH
- 53: CYAN_STROBE_FLASH
- 54: PURPLE_STROBE_FLASH
- 55: WHITE_STROBE_FLASH
- 56: SEVEN_COLOR_JUMPING_CHANGE

Speed values: 1 (fastest) to 255 (slowest)

## Architecture

The API follows a stateless design:

1. Each request discovers or connects to the device fresh
2. Command is executed
3. Device connection is closed
4. Response is returned

This ensures no connection pooling or state management is needed, though it may be slightly slower than maintaining persistent connections.

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `404`: Device not found
- `422`: Validation error (invalid parameters)
- `500`: Internal server error
- `503`: Device communication failure
