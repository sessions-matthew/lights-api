"""
Kasa Device Wrapper

Wrapper for python-kasa library to match the interface of Triones and Philips controllers.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from kasa import Discover, Device, Light
    from kasa.exceptions import KasaException
    KASA_AVAILABLE = True
except ImportError as e:
    # Fallback for when kasa is not installed
    class Discover:
        @staticmethod
        async def discover(timeout=5):
            return {}
    
    class Device:
        pass
    
    class Light:
        pass
    
    class KasaException(Exception):
        pass
    
    KASA_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class KasaStatus:
    """Kasa device status"""
    is_on: bool
    brightness: Optional[int] = None  # 0-100
    red: Optional[int] = None  # 0-255
    green: Optional[int] = None  # 0-255
    blue: Optional[int] = None  # 0-255
    temperature: Optional[int] = None  # Color temperature in K
    rgb_hex: Optional[str] = None
    
class KasaController:
    """Controller for Kasa smart devices"""
    
    def __init__(self, device):
        self.device = device
        self.name = getattr(device, 'alias', getattr(device, 'model', str(device)))
        # Use MAC address as primary identifier, fall back to IP if MAC not available
        self.address = getattr(device, 'mac', getattr(device, 'host', getattr(device, '_host', 'unknown')))
        # Keep IP for actual connection
        self.host = getattr(device, 'host', getattr(device, '_host', 'unknown'))
        self._connected = False
        
    async def connect(self) -> bool:
        """Connect to the Kasa device"""
        try:
            logger.debug(f"Connecting to Kasa device {self.name} (MAC: {self.address}, IP: {self.host})")
            await self.device.update()
            self._connected = True
            logger.debug(f"Successfully connected to Kasa device {self.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kasa device {self.name} (MAC: {self.address}, IP: {self.host}): {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the Kasa device (no-op for Kasa)"""
        self._connected = False
    
    async def get_status(self) -> Optional[KasaStatus]:
        """Get current device status"""
        try:
            await self.device.update()
            
            status = KasaStatus(is_on=self.device.is_on)
            
            # Check if device has light module (new API)
            if hasattr(self.device, 'modules') and 'Light' in self.device.modules:
                light_module = self.device.modules['Light']
                
                # Get brightness (0-100)
                if hasattr(light_module, 'brightness'):
                    status.brightness = light_module.brightness
                
                # Get HSV color if available
                if hasattr(light_module, 'hsv'):
                    hsv = light_module.hsv
                    if hsv and len(hsv) >= 3:
                        h, s, v = hsv.hue, hsv.saturation, hsv.value
                        # Convert HSV to RGB
                        status.red, status.green, status.blue = self._hsv_to_rgb(h, s, v)
                        status.rgb_hex = f"#{status.red:02x}{status.green:02x}{status.blue:02x}"
                
                # Get color temperature
                if hasattr(light_module, 'color_temp'):
                    status.temperature = light_module.color_temp
            
            # Fallback for older API or direct Light device
            elif isinstance(self.device, Light):
                # Get brightness
                if hasattr(self.device, 'brightness'):
                    status.brightness = self.device.brightness
                
                # Check for color support
                if hasattr(self.device, 'hsv') and self.device.hsv:
                    hsv = self.device.hsv
                    if hasattr(hsv, 'hue'):
                        h, s, v = hsv.hue, hsv.saturation, hsv.value
                    else:
                        h, s, v = hsv
                    # Convert HSV to RGB
                    status.red, status.green, status.blue = self._hsv_to_rgb(h, s, v)
                    status.rgb_hex = f"#{status.red:02x}{status.green:02x}{status.blue:02x}"
                
                # Check for color temperature
                if hasattr(self.device, 'color_temp'):
                    status.temperature = self.device.color_temp
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get status from Kasa device {self.address}: {e}")
            return None
    
    async def power_on(self) -> bool:
        """Turn device on"""
        try:
            await self.device.turn_on()
            return True
        except Exception as e:
            logger.error(f"Failed to turn on Kasa device {self.address}: {e}")
            return False
    
    async def power_off(self) -> bool:
        """Turn device off"""
        try:
            await self.device.turn_off()
            return True
        except Exception as e:
            logger.error(f"Failed to turn off Kasa device {self.address}: {e}")
            return False
    
    async def set_brightness(self, brightness: int) -> bool:
        """Set brightness (0-100)"""
        try:
            # Try new API with modules first
            if hasattr(self.device, 'modules') and 'Light' in self.device.modules:
                light_module = self.device.modules['Light']
                if hasattr(light_module, 'set_brightness'):
                    await light_module.set_brightness(brightness)
                    return True
            # Fallback to direct method
            elif hasattr(self.device, 'set_brightness'):
                await self.device.set_brightness(brightness)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to set brightness on Kasa device {self.address}: {e}")
            return False
    
    async def set_rgb(self, red: int, green: int, blue: int) -> bool:
        """Set RGB color (0-255 each)"""
        try:
            # Convert RGB to HSV
            h, s, v = self._rgb_to_hsv(red, green, blue)
            
            # Try new API with modules first
            if hasattr(self.device, 'modules') and 'Light' in self.device.modules:
                light_module = self.device.modules['Light']
                if hasattr(light_module, 'set_hsv'):
                    from kasa import HSV
                    await light_module.set_hsv(HSV(hue=h, saturation=s, value=v))
                    return True
            # Fallback to direct method
            elif hasattr(self.device, 'set_hsv'):
                await self.device.set_hsv(hue=h, saturation=s, value=v)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to set RGB on Kasa device {self.address}: {e}")
            return False
    
    async def set_color_hex(self, hex_color: str) -> bool:
        """Set color using hex string"""
        try:
            # Remove # if present
            hex_color = hex_color.lstrip('#')
            if len(hex_color) != 6:
                return False
            
            red = int(hex_color[0:2], 16)
            green = int(hex_color[2:4], 16)
            blue = int(hex_color[4:6], 16)
            
            return await self.set_rgb(red, green, blue)
        except Exception as e:
            logger.error(f"Failed to set hex color on Kasa device {self.address}: {e}")
            return False
    
    async def set_color_temp(self, temperature: int) -> bool:
        """Set color temperature in Kelvin"""
        try:
            # Try new API with modules first
            if hasattr(self.device, 'modules') and 'Light' in self.device.modules:
                light_module = self.device.modules['Light']
                if hasattr(light_module, 'set_color_temp'):
                    await light_module.set_color_temp(temperature)
                    return True
            # Fallback to direct method
            elif hasattr(self.device, 'set_color_temp'):
                await self.device.set_color_temp(temperature)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to set color temperature on Kasa device {self.address}: {e}")
            return False
    
    def _rgb_to_hsv(self, r: int, g: int, b: int) -> tuple:
        """Convert RGB (0-255) to HSV (0-360, 0-100, 0-100)"""
        r, g, b = r/255.0, g/255.0, b/255.0
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Hue
        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:  # max_val == b
            h = (60 * ((r - g) / diff) + 240) % 360
        
        # Saturation
        s = 0 if max_val == 0 else (diff / max_val) * 100
        
        # Value
        v = max_val * 100
        
        return int(h), int(s), int(v)
    
    def _hsv_to_rgb(self, h: int, s: int, v: int) -> tuple:
        """Convert HSV (0-360, 0-100, 0-100) to RGB (0-255)"""
        h, s, v = h/360.0, s/100.0, v/100.0
        
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        
        if 0 <= h < 1/6:
            r, g, b = c, x, 0
        elif 1/6 <= h < 2/6:
            r, g, b = x, c, 0
        elif 2/6 <= h < 3/6:
            r, g, b = 0, c, x
        elif 3/6 <= h < 4/6:
            r, g, b = 0, x, c
        elif 4/6 <= h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)

class KasaScanner:
    """Scanner for Kasa devices"""
    
    @staticmethod
    async def discover(timeout: float = 5.0) -> List[KasaController]:
        """Discover Kasa devices on the network"""
        if not KASA_AVAILABLE:
            logger.debug("Kasa library not available, skipping discovery")
            return []
            
        try:
            logger.info(f"Starting Kasa device discovery with timeout {timeout}s")
            devices = await Discover.discover(timeout=timeout)
            logger.info(f"Kasa discovery found {len(devices)} total devices")
            
            controllers = []
            
            for ip, device in devices.items():
                try:
                    # Update device info to get capabilities
                    await device.update()
                    
                    # Get MAC address for consistent identification
                    mac_address = getattr(device, 'mac', ip)
                    device_name = getattr(device, 'alias', 'Unknown Device')
                    
                    # Check if it's a light device by looking for common light features
                    is_light = False
                    
                    # Check if it's explicitly a Light device
                    if isinstance(device, Light):
                        is_light = True
                        logger.debug(f"Device {device_name} ({mac_address}) is Light device")
                    
                    # Check for Light module (new API)
                    elif hasattr(device, 'modules') and 'Light' in device.modules:
                        is_light = True
                        logger.debug(f"Device {device_name} ({mac_address}) has Light module")
                    
                    # Check device type string if available
                    elif hasattr(device, 'device_type'):
                        device_type = str(device.device_type).lower()
                        if any(light_type in device_type for light_type in ['bulb', 'light', 'strip']):
                            is_light = True
                            logger.debug(f"Device {device_name} ({mac_address}) type: {device.device_type}")
                    
                    # Check for light-like features (fallback)
                    elif (hasattr(device, 'brightness') or 
                          hasattr(device, 'set_brightness') or
                          hasattr(device, 'hsv') or
                          hasattr(device, 'color_temp')):
                        is_light = True
                        logger.debug(f"Device {device_name} ({mac_address}) has light features")
                    
                    if is_light:
                        controller = KasaController(device)
                        controllers.append(controller)
                        logger.info(f"Added Kasa light device: {device_name} (MAC: {controller.address}, IP: {controller.host})")
                    else:
                        logger.debug(f"Skipping non-light device: {device_name} ({mac_address}) - type: {getattr(device, 'device_type', 'unknown')}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process Kasa device at {ip}: {e}")
                    continue
            
            logger.info(f"Discovered {len(controllers)} Kasa light devices")
            return controllers
            
        except Exception as e:
            logger.error(f"Kasa discovery failed: {e}")
            return []

# Debug function for testing Kasa discovery
async def test_kasa_discovery():
    """Test function for Kasa device discovery - for debugging only"""
    print("Testing Kasa device discovery...")
    print(f"Kasa library available: {KASA_AVAILABLE}")
    
    if not KASA_AVAILABLE:
        print("Kasa library not installed. Install with: pip install python-kasa")
        return
    
    try:
        print("Starting discovery...")
        devices = await KasaScanner.discover(timeout=10.0)
        print(f"Found {len(devices)} Kasa light devices:")
        
        for device in devices:
            print(f"  - {device.name}")
            print(f"    MAC: {device.address}")
            print(f"    IP: {device.host}")
            if await device.connect():
                status = await device.get_status()
                print(f"    Status: {status}")
                await device.disconnect()
            else:
                print(f"    Failed to connect")
    
    except Exception as e:
        print(f"Discovery test failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_kasa_discovery())