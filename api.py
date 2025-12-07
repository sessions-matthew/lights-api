"""
REST API Backend for Triones Controller

A FastAPI-based REST API for controlling Triones RGBW Bluetooth LED controllers.
Devices are connected per-request and not persisted between requests.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
import logging
import sys
from pathlib import Path

# Add bleak-triones-controller to path
sys.path.insert(0, str(Path(__file__).parent / "bleak-triones-controller"))

from triones import TrionesController, TrionesScanner, TrionesStatus, TrionesMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Triones Controller API",
    description="REST API for controlling Triones RGBW Bluetooth LED controllers",
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

class DeviceStatus(BaseModel):
    """Device status response"""
    is_on: bool
    mode: int
    speed: int
    red: int
    green: int
    blue: int
    white: int
    rgb_hex: str
    
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

class SuccessResponse(BaseModel):
    """Success response"""
    success: bool
    message: str

# Helper Functions
async def get_controller(address: str) -> TrionesController:
    """
    Get a controller by address and connect to it.
    
    Args:
        address: Device MAC address
        
    Returns:
        TrionesController: Connected controller
        
    Raises:
        HTTPException: If device not found or connection fails
    """
    try:
        # Discover devices
        devices = await TrionesScanner.discover(timeout=5.0)
        
        # Find device by address
        for controller in devices:
            if controller.address.lower() == address.lower():
                # Connect to device
                if await controller.connect():
                    return controller
                else:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Failed to connect to device {address}"
                    )
        
        # Device not found
        raise HTTPException(
            status_code=404,
            detail=f"Device with address {address} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting controller: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def execute_command(address: str, command_func):
    """
    Execute a command on a device and handle connection cleanup.
    
    Args:
        address: Device MAC address
        command_func: Async function to execute command
        
    Returns:
        Result from command_func
    """
    controller = None
    try:
        controller = await get_controller(address)
        result = await command_func(controller)
        return result
    finally:
        if controller:
            await controller.disconnect()

# API Endpoints

@app.get("/", response_model=dict)
async def root():
    """API root endpoint"""
    return {
        "name": "Triones Controller API",
        "version": "1.0.0",
        "endpoints": {
            "discover": "/devices",
            "status": "/devices/{address}/status",
            "power_on": "/devices/{address}/power/on",
            "power_off": "/devices/{address}/power/off",
            "set_rgb": "/devices/{address}/color/rgb",
            "set_hex": "/devices/{address}/color/hex",
            "set_white": "/devices/{address}/color/white",
            "set_mode": "/devices/{address}/mode"
        }
    }

@app.get("/devices", response_model=List[DeviceInfo])
async def discover_devices(timeout: float = Query(default=10.0, ge=1.0, le=30.0)):
    """
    Discover Triones devices on the network
    
    Args:
        timeout: Discovery timeout in seconds (1-30)
        
    Returns:
        List of discovered devices
    """
    try:
        controllers = await TrionesScanner.discover(timeout=timeout)
        return [
            DeviceInfo(name=c.name, address=c.address)
            for c in controllers
        ]
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
    async def get_status(controller: TrionesController):
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
            rgb_hex=status.rgb_hex
        )
    
    return await execute_command(address, get_status)

@app.post("/devices/{address}/power/on", response_model=SuccessResponse)
async def power_on_device(address: str):
    """
    Turn device on
    
    Args:
        address: Device MAC address
        
    Returns:
        Success response
    """
    async def power_on(controller: TrionesController):
        success = await controller.power_on()
        if not success:
            raise HTTPException(status_code=503, detail="Failed to power on device")
        return SuccessResponse(success=True, message="Device powered on")
    
    return await execute_command(address, power_on)

@app.post("/devices/{address}/power/off", response_model=SuccessResponse)
async def power_off_device(address: str):
    """
    Turn device off
    
    Args:
        address: Device MAC address
        
    Returns:
        Success response
    """
    async def power_off(controller: TrionesController):
        success = await controller.power_off()
        if not success:
            raise HTTPException(status_code=503, detail="Failed to power off device")
        return SuccessResponse(success=True, message="Device powered off")
    
    return await execute_command(address, power_off)

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
    async def set_color(controller: TrionesController):
        success = await controller.set_rgb(rgb.red, rgb.green, rgb.blue)
        if not success:
            raise HTTPException(status_code=503, detail="Failed to set RGB color")
        return SuccessResponse(
            success=True,
            message=f"Color set to RGB({rgb.red}, {rgb.green}, {rgb.blue})"
        )
    
    return await execute_command(address, set_color)

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
    async def set_color(controller: TrionesController):
        success = await controller.set_color_hex(hex_color.color)
        if not success:
            raise HTTPException(status_code=503, detail="Failed to set hex color")
        return SuccessResponse(
            success=True,
            message=f"Color set to {hex_color.color}"
        )
    
    return await execute_command(address, set_color)

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
    async def set_color(controller: TrionesController):
        success = await controller.set_white(white.intensity)
        if not success:
            raise HTTPException(status_code=503, detail="Failed to set white color")
        return SuccessResponse(
            success=True,
            message=f"White intensity set to {white.intensity}"
        )
    
    return await execute_command(address, set_color)

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
    async def set_mode(controller: TrionesController):
        success = await controller.set_built_in_mode(mode_req.mode, mode_req.speed)
        if not success:
            raise HTTPException(status_code=503, detail="Failed to set mode")
        return SuccessResponse(
            success=True,
            message=f"Mode set to {mode_req.mode} with speed {mode_req.speed}"
        )
    
    return await execute_command(address, set_mode)

@app.get("/modes", response_model=dict)
async def list_modes():
    """
    List all available built-in modes
    
    Returns:
        Dictionary of mode names and values
    """
    return {
        mode.name: mode.value
        for mode in TrionesMode
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
