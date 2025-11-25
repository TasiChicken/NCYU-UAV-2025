from dataclasses import dataclass

@dataclass
class DroneConfig:
    # Movement Speeds (Used in keyboard control)
    FB_SPEED: int = 20
    LF_SPEED: int = 20
    UD_SPEED: int = 20
    DEGREE: int = 10
    
    # Safety
    MAX_RC: int = 20
