"""
REST API Backend for Local Light Controllers

A FastAPI-based REST API for controlling lights
Devices are connected per-request and not persisted between requests.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict
import asyncio
import logging
import sys
import json
from pathlib import Path
from dataclasses import dataclass

from triones import TrionesController, TrionesScanner, TrionesStatus, TrionesMode
from phillips import PhilipsController, PhilipsScanner, PhilipsStatus
from kasa_wrapper import KasaController, KasaScanner, KasaStatus

@dataclass
class DeviceCacheEntry:
    """Cache entry for tracking device across multiple scans"""
    controller: Union[TrionesController, PhilipsController, KasaController]
    scans_not_seen: int = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Light Controller API",
    description="REST API for controlling Triones, Philips BLE, and Kasa WiFi light controllers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class DeviceInfo(BaseModel):
    """Device information"""
    name: str
    address: str
    # device type: 'triones', 'philips', or 'kasa' (when known)
    type: Optional[str] = None

class DeviceStatus(BaseModel):
    """Device status response (fields optional to support multiple controllers)"""
    is_on: Optional[bool] = None
    mode: Optional[int] = None
    speed: Optional[int] = None
    red: Optional[int] = None
    green: Optional[int] = None
    blue: Optional[int] = None
    white: Optional[int] = None
    rgb_hex: Optional[str] = None
    # Philips-specific: brightness 0-255
    brightness: Optional[int] = None
    # device type
    type: Optional[str] = None
    
class RGBRequest(BaseModel):
    """RGB color request"""
    red: int = Field(ge=0, le=255, description="Red component (0-255)")
    green: int = Field(ge=0, le=255, description="Green component (0-255)")
    blue: int = Field(ge=0, le=255, description="Blue component (0-255)")

class HexColorRequest(BaseModel):
    """Hex color request"""
    color: str = Field(description="Hex color (e.g., '#FF0000' or 'FF0000')")

class WhiteRequest(BaseModel):
    """White intensity request"""
    intensity: int = Field(ge=0, le=255, description="White intensity (0-255)")

class ModeRequest(BaseModel):
    """Built-in mode request"""
    mode: int = Field(ge=37, le=56, description="Built-in mode (37-56)")
    speed: int = Field(default=1, ge=1, le=255, description="Animation speed (1=fastest, 255=slowest)")

class ColorTempRequest(BaseModel):
    """Color temperature request for Kasa devices"""
    temperature: int = Field(ge=2500, le=9000, description="Color temperature in Kelvin (2500-9000)")

class SuccessResponse(BaseModel):
    """Success response"""
    success: bool
    message: str

class DeviceGroup(BaseModel):
    """Device group model"""
    name: str = Field(description="Group name")
    device_addresses: List[str] = Field(description="List of device addresses in the group")
    description: Optional[str] = Field(None, description="Optional group description")

class CreateGroupRequest(BaseModel):
    """Create group request"""
    name: str = Field(description="Group name")
    device_addresses: List[str] = Field(description="List of device addresses to include in the group")
    description: Optional[str] = Field(None, description="Optional group description")

class UpdateGroupRequest(BaseModel):
    """Update group request"""
    device_addresses: Optional[List[str]] = Field(None, description="List of device addresses to include in the group")
    description: Optional[str] = Field(None, description="Optional group description")

class GroupActionRequest(BaseModel):
    """Group action request base class"""
    pass

class GroupRGBRequest(GroupActionRequest):
    """RGB color request for group"""
    red: int = Field(ge=0, le=255, description="Red component (0-255)")
    green: int = Field(ge=0, le=255, description="Green component (0-255)")
    blue: int = Field(ge=0, le=255, description="Blue component (0-255)")

class GroupHexColorRequest(GroupActionRequest):
    """Hex color request for group"""
    color: str = Field(description="Hex color (e.g., '#FF0000' or 'FF0000')")

# Helper Functions
async def get_controller(address: str):
    """
    Get a controller by address and connect to it.
    
    Args:
        address: Device MAC address
        
    Returns:
        Union[TrionesController, PhilipsController]: Connected controller instance
        
    Raises:
        HTTPException: If device not found or connection fails
    """
    try:
        # Prefer cached scan results when available (background scanner updates this)
        async with _cache_lock:
            cached_controllers = [entry.controller for entry in _device_cache.values()]
        
        for controller in cached_controllers:
            if controller.address.lower() == address.lower():
                if await controller.connect():
                    # Update cache with connected controller to preserve services/state
                    async with _cache_lock:
                        addr_lower = address.lower()
                        if addr_lower in _device_cache:
                            _device_cache[addr_lower].controller = controller
                            _device_cache[addr_lower].scans_not_seen = 0
                    return controller
                else:
                    raise HTTPException(status_code=503, detail=f"Failed to connect to device {address}")

        # Fallback: run a quick discovery if cache missed it
        triones_devices = await TrionesScanner.discover(timeout=5.0)
        for controller in triones_devices:
            if controller.address.lower() == address.lower():
                if await controller.connect():
                    # Add newly connected device to cache
                    async with _cache_lock:
                        addr_lower = address.lower()
                        _device_cache[addr_lower] = DeviceCacheEntry(controller=controller)
                    return controller
                else:
                    raise HTTPException(status_code=503, detail=f"Failed to connect to device {address}")

        philips_devices = await PhilipsScanner.discover(timeout=5.0)
        for controller in philips_devices:
            if controller.address.lower() == address.lower():
                if await controller.connect():
                    # Add newly connected device to cache
                    async with _cache_lock:
                        addr_lower = address.lower()
                        _device_cache[addr_lower] = DeviceCacheEntry(controller=controller)
                    return controller
                else:
                    raise HTTPException(status_code=503, detail=f"Failed to connect to device {address}")

        kasa_devices = await KasaScanner.discover(timeout=5.0)
        for controller in kasa_devices:
            if controller.address.lower() == address.lower():
                if await controller.connect():
                    # Add newly connected device to cache
                    async with _cache_lock:
                        addr_lower = address.lower()
                        _device_cache[addr_lower] = DeviceCacheEntry(controller=controller)
                    return controller
                else:
                    raise HTTPException(status_code=503, detail=f"Failed to connect to device {address}")

        # Device not found
        raise HTTPException(status_code=404, detail=f"Device with address {address} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting controller: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def execute_command(address: str, command_func, command_type: str = "default"):
    """
    Execute a command on a device and handle connection cleanup.
    
    Args:
        address: Device MAC address
        command_func: Async function to execute command
        command_type: Type of command for specialized handling
        
    Returns:
        Result from command_func
    """
    controller = None
    try:
        controller = await get_controller(address)
        result = await command_func(controller)
        
        # For Triones controllers, add delay before disconnect to ensure command processing
        if isinstance(controller, TrionesController):
            # Different delays based on command complexity
            if command_type in ["color", "mode", "preset"]:
                # Color and mode commands need more time to process
                await asyncio.sleep(0.2)
            elif command_type in ["power", "status"]:
                # Power and status commands are quick but still need some delay
                await asyncio.sleep(0.1)
            else:
                # Default delay for other commands
                await asyncio.sleep(0.15)
            
        # Update cache with controller after successful command execution
        # This preserves any discovered services or connection state
        async with _cache_lock:
            addr_lower = address.lower()
            if addr_lower in _device_cache:
                _device_cache[addr_lower].controller = controller
                _device_cache[addr_lower].scans_not_seen = 0
        
        return result
    finally:
        if controller:
            await controller.disconnect()


# ----- Device cache / background scanner -----
SCAN_INTERVAL = 600  # 10 minutes
MAX_SCANS_NOT_SEEN = 10  # Remove devices after 10 scans without being seen
_device_cache: Dict[str, DeviceCacheEntry] = {}  # address -> DeviceCacheEntry
_cache_lock = asyncio.Lock()
_cache_task: Optional[asyncio.Task] = None

# ----- Device Groups -----
GROUPS_FILE = Path("device_groups.json")
_groups: Dict[str, DeviceGroup] = {}  # group_name -> DeviceGroup
_groups_lock = asyncio.Lock()

async def _scan_once(timeout: float = 5.0):
    """Perform a single discovery across supported scanners and update cache."""
    try:
        triones = await TrionesScanner.discover(timeout=timeout)
    except Exception as e:
        logger.debug(f"TrionesScanner.discover failed: {e}")
        triones = []

    try:
        philips = await PhilipsScanner.discover(timeout=timeout)
    except Exception as e:
        logger.debug(f"PhilipsScanner.discover failed: {e}")
        philips = []

    try:
        kasa = await KasaScanner.discover(timeout=timeout)
    except Exception as e:
        logger.debug(f"KasaScanner.discover failed: {e}")
        kasa = []

    combined: List[Union[TrionesController, PhilipsController, KasaController]] = []
    combined.extend(triones)
    combined.extend(philips)
    combined.extend(kasa)

    async with _cache_lock:
        # Create a set of currently discovered device addresses (case insensitive)
        discovered_addresses = {controller.address.lower() for controller in combined}
        
        # Increment scans_not_seen for all cached devices
        for entry in _device_cache.values():
            entry.scans_not_seen += 1
        
        # Reset scans_not_seen for devices that were discovered and add new devices
        for controller in combined:
            addr_lower = controller.address.lower()
            if addr_lower in _device_cache:
                # Reset counter for existing device
                _device_cache[addr_lower].scans_not_seen = 0
                # Update controller instance with fresh one from scanner
                _device_cache[addr_lower].controller = controller
            else:
                # Add new device
                _device_cache[addr_lower] = DeviceCacheEntry(controller=controller)
        
        # Remove devices that haven't been seen for MAX_SCANS_NOT_SEEN scans
        addresses_to_remove = [
            addr for addr, entry in _device_cache.items() 
            if entry.scans_not_seen >= MAX_SCANS_NOT_SEEN
        ]
        for addr in addresses_to_remove:
            del _device_cache[addr]
            
        active_devices = len(_device_cache)
        removed_devices = len(addresses_to_remove)
        
        logger.info(f"Device cache updated: {active_devices} active devices")
        if removed_devices > 0:
            logger.info(f"Removed {removed_devices} devices not seen for {MAX_SCANS_NOT_SEEN} scans")

async def _get_cached_devices() -> List[Union[TrionesController, PhilipsController, KasaController]]:
    async with _cache_lock:
        # return a list of controller instances from cache entries
        return [entry.controller for entry in _device_cache.values()]

# ----- Group Management Functions -----

def _load_groups_from_file():
    """Load device groups from file"""
    global _groups
    if GROUPS_FILE.exists():
        try:
            with open(GROUPS_FILE, 'r') as f:
                groups_data = json.load(f)
                _groups = {name: DeviceGroup(**data) for name, data in groups_data.items()}
                logger.info(f"Loaded {len(_groups)} device groups from {GROUPS_FILE}")
        except Exception as e:
            logger.error(f"Error loading groups from {GROUPS_FILE}: {e}")
            _groups = {}
    else:
        _groups = {}
        logger.info(f"Groups file {GROUPS_FILE} not found, starting with empty groups")

def _save_groups_to_file():
    """Save device groups to file"""
    try:
        groups_data = {name: group.dict() for name, group in _groups.items()}
        with open(GROUPS_FILE, 'w') as f:
            json.dump(groups_data, f, indent=2)
        logger.info(f"Saved {len(_groups)} device groups to {GROUPS_FILE}")
    except Exception as e:
        logger.error(f"Error saving groups to {GROUPS_FILE}: {e}")
        raise

async def _get_group(group_name: str) -> DeviceGroup:
    """Get a group by name"""
    async with _groups_lock:
        if group_name not in _groups:
            raise HTTPException(status_code=404, detail=f"Group '{group_name}' not found")
        return _groups[group_name]

async def _execute_group_command(group_name: str, command_func, command_type: str = "default"):
    """Execute a command on all devices in a group"""
    group = await _get_group(group_name)
    results = []
    errors = []
    
    for address in group.device_addresses:
        try:
            result = await execute_command(address, command_func, command_type)
            results.append({"address": address, "result": result})
        except Exception as e:
            errors.append({"address": address, "error": str(e)})
            logger.warning(f"Failed to execute command on device {address}: {e}")
    
    return {
        "group_name": group_name,
        "total_devices": len(group.device_addresses),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }

async def _scan_loop():
    # initial short delay before first background loop iteration is handled by startup
    while True:
        try:
            await _scan_once()
        except Exception as e:
            logger.error(f"Background scan failed: {e}")
        await asyncio.sleep(SCAN_INTERVAL)


@app.on_event("startup")
async def _startup_scanner():
    """Start background scanner and perform an initial scan."""
    # Load device groups from file
    _load_groups_from_file()
    
    # perform an initial scan synchronously so `/devices` has values immediately
    try:
        await _scan_once()
    except Exception as e:
        logger.warning(f"Initial device scan failed: {e}")

    global _cache_task
    if _cache_task is None:
        _cache_task = asyncio.create_task(_scan_loop())


@app.on_event("shutdown")
async def _shutdown_scanner():
    global _cache_task
    if _cache_task:
        _cache_task.cancel()
        try:
            await _cache_task
        except asyncio.CancelledError:
            pass
        _cache_task = None

# API Endpoints

@app.get("/", response_model=dict)
async def root():
    """API root endpoint"""
    return {
        "name": "Light Controller API",
        "version": "1.0.0",
        "endpoints": {
            "discover": "/devices",
            "status": "/devices/{address}/status",
            "power_on": "/devices/{address}/power/on",
            "power_off": "/devices/{address}/power/off",
            "set_rgb": "/devices/{address}/color/rgb",
            "set_hex": "/devices/{address}/color/hex",
            "set_white": "/devices/{address}/color/white",
            "set_color_temp": "/devices/{address}/color/temperature",
            "set_mode": "/devices/{address}/mode",
            "pair": "/devices/{address}/pair",
            "groups": "/groups",
            "create_group": "/groups",
            "get_group": "/groups/{group_name}",
            "update_group": "/groups/{group_name}",
            "delete_group": "/groups/{group_name}",
            "group_power_on": "/groups/{group_name}/power/on",
            "group_power_off": "/groups/{group_name}/power/off",
            "group_set_rgb": "/groups/{group_name}/color/rgb",
            "group_set_hex": "/groups/{group_name}/color/hex"
        }
    }

@app.get("/devices", response_model=List[DeviceInfo])
async def discover_devices(timeout: float = Query(default=10.0, ge=1.0, le=30.0)):
    """
    Discover Triones, Philips, and Kasa devices on the network
    
    Args:
        timeout: Discovery timeout in seconds (1-30)
        
    Returns:
        List of discovered devices
    """
    try:
        # Prefer cached devices populated by background scanner
        cached = await _get_cached_devices()
        if cached:
            results = []
            for c in cached:
                if isinstance(c, TrionesController):
                    dtype = "triones"
                elif isinstance(c, PhilipsController):
                    dtype = "philips"
                elif isinstance(c, KasaController):
                    dtype = "kasa"
                else:
                    dtype = "unknown"
                results.append(DeviceInfo(name=c.name, address=c.address, type=dtype))
            return results

        # Cache empty? perform an immediate discovery using provided timeout
        triones_controllers = await TrionesScanner.discover(timeout=timeout)
        philips_controllers = await PhilipsScanner.discover(timeout=timeout)
        kasa_controllers = await KasaScanner.discover(timeout=timeout)
        results = []
        for c in triones_controllers:
            results.append(DeviceInfo(name=c.name, address=c.address, type="triones"))
        for c in philips_controllers:
            results.append(DeviceInfo(name=c.name, address=c.address, type="philips"))
        for c in kasa_controllers:
            results.append(DeviceInfo(name=c.name, address=c.address, type="kasa"))
        return results
    except Exception as e:
        logger.error(f"Error discovering devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/devices/{address}/status", response_model=DeviceStatus)
async def get_device_status(address: str):
    """
    Get current status of a device
    
    Args:
        address: Device MAC address
        
    Returns:
        Device status information
    """
    async def get_status(controller):
        # Triones exposes a detailed status object
        if isinstance(controller, TrionesController):
            status = await controller.get_status()
            if status is None:
                raise HTTPException(status_code=503, detail="Failed to get device status")
            return DeviceStatus(
                is_on=status.is_on,
                mode=status.mode,
                speed=status.speed,
                red=status.red,
                green=status.green,
                blue=status.blue,
                white=status.white,
                rgb_hex=status.rgb_hex,
                type="triones"
            )

        # Philips exposes only power/brightness
        if isinstance(controller, PhilipsController):
            p = await controller.read_power()
            b = await controller.read_brightness()
            is_on = None
            if p is not None:
                try:
                    is_on = bool(int(p))
                except Exception:
                    is_on = None
            return DeviceStatus(is_on=is_on, brightness=b, type="philips")

        # Kasa exposes comprehensive status
        if isinstance(controller, KasaController):
            status = await controller.get_status()
            if status is None:
                raise HTTPException(status_code=503, detail="Failed to get device status")
            return DeviceStatus(
                is_on=status.is_on,
                brightness=status.brightness,
                red=status.red,
                green=status.green,
                blue=status.blue,
                rgb_hex=status.rgb_hex,
                type="kasa"
            )

        raise HTTPException(status_code=500, detail="Unknown controller type")
    
    return await execute_command(address, get_status, "status")

@app.post("/devices/{address}/power/on", response_model=SuccessResponse)
async def power_on_device(address: str):
    """
    Turn device on
    
    Args:
        address: Device MAC address
        
    Returns:
        Success response
    """
    async def power_on(controller):
        # Triones
        if isinstance(controller, TrionesController) and hasattr(controller, "power_on"):
            success = await controller.power_on()
            if not success:
                raise HTTPException(status_code=503, detail="Failed to power on device")
            return SuccessResponse(success=True, message="Device powered on")

        # Philips: use set_power(0x01)
        if isinstance(controller, PhilipsController) and hasattr(controller, "set_power"):
            success = await controller.set_power(0x01)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to power on device")
            return SuccessResponse(success=True, message="Device powered on (philips)")

        # Kasa: use power_on
        if isinstance(controller, KasaController) and hasattr(controller, "power_on"):
            success = await controller.power_on()
            if not success:
                raise HTTPException(status_code=503, detail="Failed to power on device")
            return SuccessResponse(success=True, message="Device powered on (kasa)")

        raise HTTPException(status_code=501, detail="Power on not supported for this device")
    
    return await execute_command(address, power_on, "power")

@app.post("/devices/{address}/power/off", response_model=SuccessResponse)
async def power_off_device(address: str):
    """
    Turn device off
    
    Args:
        address: Device MAC address
        
    Returns:
        Success response
    """
    async def power_off(controller):
        if isinstance(controller, TrionesController) and hasattr(controller, "power_off"):
            success = await controller.power_off()
            if not success:
                raise HTTPException(status_code=503, detail="Failed to power off device")
            return SuccessResponse(success=True, message="Device powered off")

        if isinstance(controller, PhilipsController) and hasattr(controller, "set_power"):
            success = await controller.set_power(0x00)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to power off device")
            return SuccessResponse(success=True, message="Device powered off (philips)")

        if isinstance(controller, KasaController) and hasattr(controller, "power_off"):
            success = await controller.power_off()
            if not success:
                raise HTTPException(status_code=503, detail="Failed to power off device")
            return SuccessResponse(success=True, message="Device powered off (kasa)")

        raise HTTPException(status_code=501, detail="Power off not supported for this device")
    
    return await execute_command(address, power_off, "power")


@app.get("/devices/{address}/pair", response_model=SuccessResponse)
async def pair_device(address: str):
    """
    Pair a Philips device (if supported by platform/Bleak).

    Args:
        address: Device MAC address

    Returns:
        Success response
    """
    async def do_pair(controller):
        # Philips pairing
        if isinstance(controller, PhilipsController) and hasattr(controller, "pair"):
            paired = await controller.pair()
            if not paired:
                raise HTTPException(status_code=503, detail="Failed to pair device")
            return SuccessResponse(success=True, message="Device paired (philips)")

        # Triones typically do not require pairing via BLE
        if isinstance(controller, TrionesController):
            raise HTTPException(status_code=501, detail="Pairing not required for this device")

        # Kasa devices typically do not require pairing via the API (they use WiFi)
        if isinstance(controller, KasaController):
            raise HTTPException(status_code=501, detail="Pairing not required for this device (use Kasa app for initial setup)")

        raise HTTPException(status_code=500, detail="Unknown controller type")

    return await execute_command(address, do_pair, "pairing")

@app.post("/devices/{address}/color/rgb", response_model=SuccessResponse)
async def set_rgb_color(address: str, rgb: RGBRequest):
    """
    Set RGB color
    
    Args:
        address: Device MAC address
        rgb: RGB color values
        
    Returns:
        Success response
    """
    async def set_color(controller):
        if isinstance(controller, TrionesController) and hasattr(controller, "set_rgb"):
            success = await controller.set_rgb(rgb.red, rgb.green, rgb.blue)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set RGB color")
            return SuccessResponse(success=True, message=f"Color set to RGB({rgb.red}, {rgb.green}, {rgb.blue})")

        if isinstance(controller, KasaController) and hasattr(controller, "set_rgb"):
            success = await controller.set_rgb(rgb.red, rgb.green, rgb.blue)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set RGB color")
            return SuccessResponse(success=True, message=f"Color set to RGB({rgb.red}, {rgb.green}, {rgb.blue}) (kasa)")

        raise HTTPException(status_code=501, detail="RGB color not supported for this device")
    
    return await execute_command(address, set_color, "color")

@app.post("/devices/{address}/color/hex", response_model=SuccessResponse)
async def set_hex_color(address: str, hex_color: HexColorRequest):
    """
    Set color using hex string
    
    Args:
        address: Device MAC address
        hex_color: Hex color string
        
    Returns:
        Success response
    """
    async def set_color(controller):
        if isinstance(controller, TrionesController) and hasattr(controller, "set_color_hex"):
            success = await controller.set_color_hex(hex_color.color)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set hex color")
            return SuccessResponse(success=True, message=f"Color set to {hex_color.color}")

        if isinstance(controller, KasaController) and hasattr(controller, "set_color_hex"):
            success = await controller.set_color_hex(hex_color.color)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set hex color")
            return SuccessResponse(success=True, message=f"Color set to {hex_color.color} (kasa)")

        raise HTTPException(status_code=501, detail="Hex color not supported for this device")
    
    return await execute_command(address, set_color, "color")

@app.post("/devices/{address}/color/white", response_model=SuccessResponse)
async def set_white_color(address: str, white: WhiteRequest):
    """
    Set white color mode
    
    Args:
        address: Device MAC address
        white: White intensity
        
    Returns:
        Success response
    """
    async def set_color(controller):
        # Triones white control
        if isinstance(controller, TrionesController) and hasattr(controller, "set_white"):
            success = await controller.set_white(white.intensity)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set white color")
            return SuccessResponse(success=True, message=f"White intensity set to {white.intensity}")

        # Philips: use brightness if available
        if isinstance(controller, PhilipsController) and hasattr(controller, "set_brightness"):
            success = await controller.set_brightness(white.intensity)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set brightness")
            return SuccessResponse(success=True, message=f"Brightness set to {white.intensity} (philips)")

        # Kasa: use brightness (convert 0-255 to 0-100)
        if isinstance(controller, KasaController) and hasattr(controller, "set_brightness"):
            brightness_percent = int((white.intensity / 255) * 100)
            success = await controller.set_brightness(brightness_percent)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set brightness")
            return SuccessResponse(success=True, message=f"Brightness set to {brightness_percent}% (kasa)")

        raise HTTPException(status_code=501, detail="White/brightness not supported for this device")
    
    return await execute_command(address, set_color, "color")

@app.post("/devices/{address}/color/temperature", response_model=SuccessResponse)
async def set_color_temperature(address: str, temp_req: ColorTempRequest):
    """
    Set color temperature (Kasa devices only)
    
    Args:
        address: Device MAC address
        temp_req: Color temperature in Kelvin
        
    Returns:
        Success response
    """
    async def set_temp(controller):
        if isinstance(controller, KasaController) and hasattr(controller, "set_color_temp"):
            success = await controller.set_color_temp(temp_req.temperature)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set color temperature")
            return SuccessResponse(success=True, message=f"Color temperature set to {temp_req.temperature}K (kasa)")

        raise HTTPException(status_code=501, detail="Color temperature not supported for this device")
    
    return await execute_command(address, set_temp, "color")

@app.post("/devices/{address}/mode", response_model=SuccessResponse)
async def set_built_in_mode(address: str, mode_req: ModeRequest):
    """
    Set built-in lighting mode
    
    Args:
        address: Device MAC address
        mode_req: Mode and speed settings
        
    Returns:
        Success response
    """
    async def set_mode(controller):
        if isinstance(controller, TrionesController) and hasattr(controller, "set_built_in_mode"):
            success = await controller.set_built_in_mode(mode_req.mode, mode_req.speed)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set mode")
            return SuccessResponse(success=True, message=f"Mode set to {mode_req.mode} with speed {mode_req.speed}")

        raise HTTPException(status_code=501, detail="Built-in modes not supported for this device")
    
    return await execute_command(address, set_mode, "mode")

# ----- Device Group Endpoints -----

@app.get("/groups", response_model=Dict[str, DeviceGroup])
async def list_groups():
    """
    List all device groups
    
    Returns:
        Dictionary of group names and their configurations
    """
    async with _groups_lock:
        return _groups.copy()

@app.post("/groups", response_model=SuccessResponse)
async def create_group(group_req: CreateGroupRequest):
    """
    Create a new device group

    Args:
        group_req: Group creation request
        
    Returns:
        Success response
    """
    # Validate that all device addresses exist (outside of groups lock)
    cached_devices = await _get_cached_devices()
    cached_addresses = {c.address.lower() for c in cached_devices}
    
    for address in group_req.device_addresses:
        if address.lower() not in cached_addresses:
            logger.warning(f"Device {address} not found in cache, but allowing group creation")
    
    async with _groups_lock:
        if group_req.name in _groups:
            raise HTTPException(status_code=409, detail=f"Group '{group_req.name}' already exists")
        
        _groups[group_req.name] = DeviceGroup(
            name=group_req.name,
            device_addresses=group_req.device_addresses,
            description=group_req.description
        )
        
        _save_groups_to_file()
        
        return SuccessResponse(
            success=True, 
            message=f"Group '{group_req.name}' created with {len(group_req.device_addresses)} devices"
        )@app.get("/groups/{group_name}", response_model=DeviceGroup)
async def get_group(group_name: str):
    """
    Get a specific device group
    
    Args:
        group_name: Name of the group
        
    Returns:
        Device group information
    """
    return await _get_group(group_name)

@app.put("/groups/{group_name}", response_model=SuccessResponse)
async def update_group(group_name: str, update_req: UpdateGroupRequest):
    """
    Update a device group

    Args:
        group_name: Name of the group to update
        update_req: Update request
        
    Returns:
        Success response
    """
    # Validate device addresses if provided (outside of groups lock)
    if update_req.device_addresses is not None:
        cached_devices = await _get_cached_devices()
        cached_addresses = {c.address.lower() for c in cached_devices}
        
        for address in update_req.device_addresses:
            if address.lower() not in cached_addresses:
                logger.warning(f"Device {address} not found in cache, but allowing group update")
    
    async with _groups_lock:
        if group_name not in _groups:
            raise HTTPException(status_code=404, detail=f"Group '{group_name}' not found")
        
        group = _groups[group_name]
        
        if update_req.device_addresses is not None:
            group.device_addresses = update_req.device_addresses
        
        if update_req.description is not None:
            group.description = update_req.description
        
        _save_groups_to_file()
        
        return SuccessResponse(
            success=True,
            message=f"Group '{group_name}' updated"
        )@app.delete("/groups/{group_name}", response_model=SuccessResponse)
async def delete_group(group_name: str):
    """
    Delete a device group
    
    Args:
        group_name: Name of the group to delete
        
    Returns:
        Success response
    """
    async with _groups_lock:
        if group_name not in _groups:
            raise HTTPException(status_code=404, detail=f"Group '{group_name}' not found")
        
        del _groups[group_name]
        _save_groups_to_file()
        
        return SuccessResponse(
            success=True,
            message=f"Group '{group_name}' deleted"
        )

# ----- Group Control Endpoints -----

@app.post("/groups/{group_name}/power/on", response_model=dict)
async def group_power_on(group_name: str):
    """
    Turn on all devices in a group
    
    Args:
        group_name: Name of the group
        
    Returns:
        Group operation results
    """
    async def power_on(controller):
        success = await controller.turn_on()
        if not success:
            raise HTTPException(status_code=503, detail="Failed to turn on device")
        return SuccessResponse(success=True, message="Device turned on")
    
    return await _execute_group_command(group_name, power_on, "power")

@app.post("/groups/{group_name}/power/off", response_model=dict)
async def group_power_off(group_name: str):
    """
    Turn off all devices in a group
    
    Args:
        group_name: Name of the group
        
    Returns:
        Group operation results
    """
    async def power_off(controller):
        success = await controller.turn_off()
        if not success:
            raise HTTPException(status_code=503, detail="Failed to turn off device")
        return SuccessResponse(success=True, message="Device turned off")
    
    return await _execute_group_command(group_name, power_off, "power")

@app.post("/groups/{group_name}/color/rgb", response_model=dict)
async def group_set_rgb_color(group_name: str, rgb_req: GroupRGBRequest):
    """
    Set RGB color for all devices in a group
    
    Args:
        group_name: Name of the group
        rgb_req: RGB color values
        
    Returns:
        Group operation results
    """
    async def set_rgb(controller):
        if isinstance(controller, TrionesController) and hasattr(controller, "set_rgb"):
            success = await controller.set_rgb(rgb_req.red, rgb_req.green, rgb_req.blue)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set RGB color")
            return SuccessResponse(success=True, message=f"RGB set to ({rgb_req.red}, {rgb_req.green}, {rgb_req.blue}) (triones)")
        elif isinstance(controller, PhilipsController) and hasattr(controller, "set_rgb"):
            success = await controller.set_rgb(rgb_req.red, rgb_req.green, rgb_req.blue)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set RGB color")
            return SuccessResponse(success=True, message=f"RGB set to ({rgb_req.red}, {rgb_req.green}, {rgb_req.blue}) (philips)")
        elif isinstance(controller, KasaController) and hasattr(controller, "set_hsv"):
            import colorsys
            h, s, v = colorsys.rgb_to_hsv(rgb_req.red/255.0, rgb_req.green/255.0, rgb_req.blue/255.0)
            success = await controller.set_hsv(int(h*360), int(s*100), int(v*100))
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set RGB color")
            return SuccessResponse(success=True, message=f"RGB set to ({rgb_req.red}, {rgb_req.green}, {rgb_req.blue}) (kasa)")
        
        raise HTTPException(status_code=501, detail="RGB color not supported for this device")
    
    return await _execute_group_command(group_name, set_rgb, "color")

@app.post("/groups/{group_name}/color/hex", response_model=dict)
async def group_set_hex_color(group_name: str, hex_req: GroupHexColorRequest):
    """
    Set hex color for all devices in a group
    
    Args:
        group_name: Name of the group
        hex_req: Hex color value
        
    Returns:
        Group operation results
    """
    # Convert hex to RGB
    hex_color = hex_req.color.lstrip('#')
    if len(hex_color) != 6:
        raise HTTPException(status_code=400, detail="Invalid hex color format. Use '#RRGGBB' or 'RRGGBB'")
    
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid hex color format")
    
    async def set_hex(controller):
        if isinstance(controller, TrionesController) and hasattr(controller, "set_rgb"):
            success = await controller.set_rgb(r, g, b)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set hex color")
            return SuccessResponse(success=True, message=f"Hex color set to {hex_req.color} (triones)")
        elif isinstance(controller, PhilipsController) and hasattr(controller, "set_rgb"):
            success = await controller.set_rgb(r, g, b)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set hex color")
            return SuccessResponse(success=True, message=f"Hex color set to {hex_req.color} (philips)")
        elif isinstance(controller, KasaController) and hasattr(controller, "set_hsv"):
            import colorsys
            h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            success = await controller.set_hsv(int(h*360), int(s*100), int(v*100))
            if not success:
                raise HTTPException(status_code=503, detail="Failed to set hex color")
            return SuccessResponse(success=True, message=f"Hex color set to {hex_req.color} (kasa)")
        
        raise HTTPException(status_code=501, detail="Hex color not supported for this device")
    
    return await _execute_group_command(group_name, set_hex, "color")

@app.get("/modes", response_model=dict)
async def list_modes():
    """
    List all available built-in modes
    
    Returns:
        Dictionary of mode names and values
    """
    # Only Triones exposes built-in modes; return empty mapping for Philips
    return {mode.name: mode.value for mode in TrionesMode}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
