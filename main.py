"""
VRM Action MCP Server
Handles different VRM model capabilities and generates appropriate actions
"""

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union

mcp = FastMCP("VRM_ActionServer")

class VRMCapabilities(BaseModel):
    """VRM model capabilities structure"""
    expressions: List[str] = Field(default_factory=list, description="Available facial expressions")
    bones: List[str] = Field(default_factory=list, description="Available bone nodes")
    has_finger_bones: bool = Field(default=False, description="Whether model has detailed finger bones")
    has_toe_bones: bool = Field(default=False, description="Whether model has toe bones")
    has_spring_bones: bool = Field(default=False, description="Whether model has physics bones")

class BoneTransform(BaseModel):
    """Bone transformation data"""
    bone_name: str
    rotation: List[float] = Field(description="Euler rotation [x, y, z] in radians")
    position: Optional[List[float]] = Field(default=None, description="Position offset [x, y, z]")

class ExpressionData(BaseModel):
    """Expression blend shape data"""
    name: str
    value: float = Field(ge=0.0, le=1.0, description="Expression intensity 0-1")

class VRMAction(BaseModel):
    """Complete VRM action structure"""
    name: str = Field(description="Action name/description")
    duration: float = Field(default=1.0, description="Action duration in seconds")
    expressions: List[ExpressionData] = Field(default_factory=list)
    bone_transforms: List[BoneTransform] = Field(default_factory=list)
    loop: bool = Field(default=False, description="Whether action should loop")

class ActionRequest(BaseModel):
    """Request structure for generating actions"""
    action_type: str = Field(description="Type of action (dance, gesture, emotion, etc.)")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="Action intensity")
    duration: float = Field(default=2.0, description="Duration in seconds")
    model_capabilities: Optional[VRMCapabilities] = Field(default=None)

# Predefined action templates
ACTION_TEMPLATES = {
    "wave_hello": {
        "expressions": [{"name": "happy", "value": 0.7}],
        "bone_transforms": [
            {"bone_name": "rightUpperArm", "rotation": [0, 0, -1.57]},
            {"bone_name": "rightLowerArm", "rotation": [0, 0, -0.5]},
        ]
    },
    "dance_basic": {
        "expressions": [{"name": "happy", "value": 0.8}],
        "bone_transforms": [
            {"bone_name": "leftUpperArm", "rotation": [0, 0, 1.0]},
            {"bone_name": "rightUpperArm", "rotation": [0, 0, -1.0]},
            {"bone_name": "spine", "rotation": [0, 0.3, 0]},
        ]
    },
    "point_finger": {
        "bone_transforms": [
            {"bone_name": "rightUpperArm", "rotation": [0, -0.5, -1.2]},
            {"bone_name": "rightLowerArm", "rotation": [0, 0, -0.3]},
        ]
    },
    "bow": {
        "expressions": [{"name": "neutral", "value": 1.0}],
        "bone_transforms": [
            {"bone_name": "spine", "rotation": [0.8, 0, 0]},
            {"bone_name": "neck", "rotation": [0.3, 0, 0]},
        ]
    },
    "clap": {
        "expressions": [{"name": "happy", "value": 0.6}],
        "bone_transforms": [
            {"bone_name": "rightUpperArm", "rotation": [0, -0.8, -1.0]},
            {"bone_name": "leftUpperArm", "rotation": [0, 0.8, 1.0]},
            {"bone_name": "rightLowerArm", "rotation": [0, 0, -1.2]},
            {"bone_name": "leftLowerArm", "rotation": [0, 0, 1.2]},
        ]
    }
}

STANDARD_EXPRESSIONS = [
    "neutral", "happy", "angry", "sad", "relaxed", "surprised",
    "blink", "blink_l", "blink_r", "look_up", "look_down", 
    "look_left", "look_right", "a", "i", "u", "e", "o"
]

STANDARD_BONES = [
    "hips", "spine", "chest", "neck", "head",
    "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
    "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot"
]

@mcp.tool()
def get_model_capabilities(expressions: List[str] = None, bones: List[str] = None) -> VRMCapabilities:
    """
    Analyze or set VRM model capabilities.
    If no parameters provided, returns standard VRM capabilities.
    """
    if expressions is None:
        expressions = STANDARD_EXPRESSIONS
    if bones is None:
        bones = STANDARD_BONES
        
    return VRMCapabilities(
        expressions=expressions,
        bones=bones,
        has_finger_bones="leftThumb1" in bones or "rightThumb1" in bones,
        has_toe_bones="leftToes" in bones or "rightToes" in bones,
        has_spring_bones=True  # Assume spring bone support
    )

@mcp.tool()
def generate_vrm_action(request: ActionRequest) -> VRMAction:
    """
    Generate a VRM action based on the request and model capabilities.
    Adapts to the specific model's available expressions and bones.
    Uses intelligent fallbacks and alternatives based on model capabilities.
    """
    # Require model capabilities for accurate generation
    if request.model_capabilities is None:
        raise ValueError("Model capabilities must be provided for accurate action generation")
    
    capabilities = request.model_capabilities
    
    # Get base template or generate dynamic action
    template = ACTION_TEMPLATES.get(request.action_type, {})
    
    # Smart expression mapping with fallbacks
    expressions = []
    for expr_data in template.get("expressions", []):
        mapped_expr = _map_expression_with_fallback(expr_data["name"], capabilities.expressions)
        if mapped_expr:
            adjusted_value = expr_data["value"] * request.intensity
            expressions.append(ExpressionData(name=mapped_expr, value=adjusted_value))
    
    # Smart bone mapping with alternatives
    bone_transforms = []
    for bone_data in template.get("bone_transforms", []):
        mapped_bones = _map_bone_with_alternatives(bone_data["bone_name"], capabilities.bones, capabilities)
        for bone_name in mapped_bones:
            # Adjust rotation intensity and adapt for bone type
            rotation = _adapt_rotation_for_bone(bone_data["rotation"], bone_name, request.intensity)
            bone_transforms.append(BoneTransform(
                bone_name=bone_name,
                rotation=rotation,
                position=bone_data.get("position")
            ))
    
    # Generate additional actions based on unique model capabilities
    if not template:
        bone_transforms.extend(_generate_fallback_action(request, capabilities))
    
    return VRMAction(
        name=f"{request.action_type}_{request.intensity}",
        duration=request.duration,
        expressions=expressions,
        bone_transforms=bone_transforms,
        loop=request.action_type in ["dance_basic", "idle_animation"]
    )

@mcp.tool()
def list_available_actions() -> List[str]:
    """List all available action types"""
    return list(ACTION_TEMPLATES.keys())

@mcp.tool()
def create_custom_action(
    name: str,
    expressions: List[Dict] = None,
    bone_transforms: List[Dict] = None,
    duration: float = 2.0,
    loop: bool = False
) -> VRMAction:
    """
    Create a custom VRM action with specific expressions and bone transforms.
    
    Example expressions: [{"name": "happy", "value": 0.8}]
    Example bone_transforms: [{"bone_name": "rightUpperArm", "rotation": [0, 0, -1.57]}]
    """
    expr_list = []
    if expressions:
        for expr in expressions:
            expr_list.append(ExpressionData(**expr))
    
    bone_list = []
    if bone_transforms:
        for bone in bone_transforms:
            bone_list.append(BoneTransform(**bone))
    
    return VRMAction(
        name=name,
        duration=duration,
        expressions=expr_list,
        bone_transforms=bone_list,
        loop=loop
    )

@mcp.tool()
def get_action_sequence(action_names: List[str], model_capabilities: VRMCapabilities = None) -> List[VRMAction]:
    """
    Generate a sequence of VRM actions that can be played consecutively.
    """
    sequence = []
    for action_name in action_names:
        request = ActionRequest(
            action_type=action_name,
            intensity=0.7,
            model_capabilities=model_capabilities
        )
        action = generate_vrm_action(request)
        sequence.append(action)
    
    return sequence

# Smart mapping functions for model-specific adaptation
def _map_expression_with_fallback(target_expr: str, available_exprs: List[str]) -> Optional[str]:
    """Map target expression to available expressions with intelligent fallbacks"""
    if target_expr in available_exprs:
        return target_expr
    
    # Expression fallback mappings
    fallback_map = {
        "happy": ["joy", "smile", "pleased", "cheerful"],
        "sad": ["sorrow", "cry", "depressed", "down"],
        "angry": ["mad", "irritated", "upset", "furious"],
        "surprised": ["shock", "amazed", "wow", "astonished"],
        "neutral": ["default", "rest", "normal"],
        "blink": ["blink_both", "eye_close"],
        "look_left": ["eye_left", "gaze_left"],
        "look_right": ["eye_right", "gaze_right"],
        "look_up": ["eye_up", "gaze_up"],
        "look_down": ["eye_down", "gaze_down"]
    }
    
    # Try fallbacks
    for fallback in fallback_map.get(target_expr, []):
        if fallback in available_exprs:
            return fallback
    
    return None

def _map_bone_with_alternatives(target_bone: str, available_bones: List[str], capabilities: VRMCapabilities) -> List[str]:
    """Map target bone to available bones with alternatives and enhancements"""
    mapped_bones = []
    
    if target_bone in available_bones:
        mapped_bones.append(target_bone)
    
    # Bone alternative mappings
    bone_alternatives = {
        "rightUpperArm": ["rightArm", "rightShoulder"],
        "leftUpperArm": ["leftArm", "leftShoulder"],
        "rightLowerArm": ["rightForearm", "rightElbow"],
        "leftLowerArm": ["leftForearm", "leftElbow"],
        "spine": ["chest", "upperChest", "torso"],
        "neck": ["head", "spine"],
        "rightHand": ["rightWrist"],
        "leftHand": ["leftWrist"]
    }
    
    # Try alternatives if main bone not found
    if not mapped_bones:
        for alt_bone in bone_alternatives.get(target_bone, []):
            if alt_bone in available_bones:
                mapped_bones.append(alt_bone)
                break
    
    # Add finger bones if available and relevant
    if target_bone in ["rightHand", "leftHand"] and capabilities.has_finger_bones:
        finger_bones = _get_finger_bones(target_bone, available_bones)
        mapped_bones.extend(finger_bones)
    
    return mapped_bones

def _get_finger_bones(hand_bone: str, available_bones: List[str]) -> List[str]:
    """Get available finger bones for enhanced hand gestures"""
    finger_bones = []
    hand_prefix = "right" if "right" in hand_bone.lower() else "left"
    
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Little"]
    for finger in finger_names:
        for i in [1, 2, 3]:  # finger segments
            bone_name = f"{hand_prefix}{finger}{i}"
            if bone_name in available_bones:
                finger_bones.append(bone_name)
    
    return finger_bones

def _adapt_rotation_for_bone(base_rotation: List[float], bone_name: str, intensity: float) -> List[float]:
    """Adapt rotation values based on bone type and model constraints"""
    # Scale by intensity
    adapted = [r * intensity for r in base_rotation]
    
    # Bone-specific constraints and adjustments
    if "finger" in bone_name.lower() or "thumb" in bone_name.lower():
        # Fingers have limited rotation ranges
        adapted = [max(-0.5, min(0.5, r)) for r in adapted]
    elif "neck" in bone_name.lower():
        # Neck has different constraints
        adapted[0] = max(-0.8, min(0.8, adapted[0]))  # Limit pitch
        adapted[1] = max(-1.2, min(1.2, adapted[1]))  # Limit yaw
    elif "spine" in bone_name.lower():
        # Spine bending limits
        adapted[0] = max(-1.0, min(1.0, adapted[0]))
    
    return adapted

def _generate_fallback_action(request: ActionRequest, capabilities: VRMCapabilities) -> List[BoneTransform]:
    """Generate fallback actions when no template exists, using available bones creatively"""
    fallback_transforms = []
    
    action_type = request.action_type.lower()
    intensity = request.intensity
    
    # Generate based on action type keywords
    if "wave" in action_type or "hello" in action_type:
        if "rightUpperArm" in capabilities.bones:
            fallback_transforms.append(BoneTransform(
                bone_name="rightUpperArm",
                rotation=[0, 0, -1.57 * intensity]
            ))
    
    elif "dance" in action_type or "move" in action_type:
        # Use available bones for dance-like movement
        if "spine" in capabilities.bones:
            fallback_transforms.append(BoneTransform(
                bone_name="spine",
                rotation=[0, 0.3 * intensity, 0]
            ))
        for arm in ["rightUpperArm", "leftUpperArm"]:
            if arm in capabilities.bones:
                side_mult = 1 if "right" in arm else -1
                fallback_transforms.append(BoneTransform(
                    bone_name=arm,
                    rotation=[0, 0, side_mult * 1.0 * intensity]
                ))
    
    elif "point" in action_type or "indicate" in action_type:
        if "rightUpperArm" in capabilities.bones:
            fallback_transforms.append(BoneTransform(
                bone_name="rightUpperArm",
                rotation=[0, -0.5 * intensity, -1.2 * intensity]
            ))
    
    return fallback_transforms


def main():
    """Run the MCP server"""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()