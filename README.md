# Vision-Language-Action (VLA) Autonomous Robotic Manipulation

[![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20%7C%20Iron%20%7C%20Jazzy-blue?logo=ros)](https://docs.ros.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12%20%7C%203.14-green?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+%20%7C%20Metal%20MPS-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-End--to--End%20VLA%20%2B%20Modular-orange)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An ML-driven **Vision-Language-Action (VLA)** robotic manipulation system in ROS 2. The pipeline observes the environment through an active overhead camera, ingests natural language instructions (e.g., *"Pick up the red soda can, avoid the laser barrier, and deliver it to Docking Bay 1"*), grounds spatial intent using a multimodal transformer, predicts continuous robot actions, and executes closed-loop manipulation via 6-DOF numerical inverse kinematics.

**No predefined coordinates in Python lists.** The model estimates object positions and generates action deltas directly from camera photons and depth rays.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Sensors ["1. Vision & Sensor Layer (vla_perception)"]
        CamNode["vla_camera_stream<br/>(camera_stream_node.py)"]
        RGB["/camera/color/image_raw<br/>(256x256 RGB Photons)"]
        Depth["/camera/depth/image_raw<br/>(Calibrated 32FC1 Depth)"]
        Info["/camera/camera_info<br/>(Intrinsics K Matrix)"]
        CamNode --> RGB
        CamNode --> Depth
        CamNode --> Info
    end

    subgraph HumanInput ["2. Natural Language Layer"]
        CLI["Human Prompt<br/>e.g., 'Pick up the red can...'"]
    end

    subgraph PathA ["Path A: End-to-End Neural VLA Policy (vla_perception)"]
        VLANode["vla_policy_node<br/>(Octo-VLA Multimodal Policy)"]
        ViT["Patch Visual Tokenizer (16x16)"]
        LangEnc["Language Tokenizer & Embedder"]
        Transformer["Cross-Attention Multimodal Transformer"]
        ActionHead["7-DoF Continuous Action Head"]
        AttnHeatmap["/vla/attention_map<br/>(Visual Attention Heatmap)"]

        RGB --> ViT
        CLI --> LangEnc
        ViT --> Transformer
        LangEnc --> Transformer
        Transformer --> ActionHead
        Transformer --> AttnHeatmap
    end

    subgraph PathB ["Path B: Dynamic 3D Perception & Task Planner (Modular)"]
        DetNode["vla_object_detector<br/>(detector_node.py)"]
        RayCast["3D Pin-Hole Ray Back-Projection<br/>Camera Optical -> World (Z=0.06m)"]
        Planner["vla_language_task_planner<br/>(language_task_node.py)"]

        RGB --> DetNode
        Info --> DetNode
        DetNode --> RayCast
        RayCast -->|/vla/detected_objects| Planner
        CLI --> Planner
    end

    subgraph MotionPlanner ["4. Kinematics & Execution (vla_motion_planning)"]
        TrajNode["vla_trajectory_executor<br/>(trajectory_executor_node.py)"]
        DLS["Damped Least-Squares Numerical IK<br/>3x6 Jacobian Inversion (λ = 0.02)"]
        ArcGen["Collision-Avoidance Arc Synthesis<br/>Obstacle Clearance z = 0.38m"]
        TFManager["Dynamic TF2 Spatial Reparenting<br/>base_link -> tool0 -> base_link"]

        ActionHead -->|/vla/action_cmd (dx, dy, dz, grip)| TrajNode
        Planner -->|/vla/task_dispatch (JSON Waypoint Plan)| TrajNode
        TrajNode --> DLS
        TrajNode --> ArcGen
        TrajNode --> TFManager
    end

    subgraph DigitalTwin ["5. Digital Twin & Visualization (RViz2)"]
        RSP["robot_state_publisher<br/>URDF & Kinematic Meshes"]
        RViz["RViz2 Studio Viewport<br/>Light Studio Slate (215, 220, 228)"]
    end

    TrajNode -->|/joint_states @ 50Hz| RSP
    RSP -->|/tf & /robot_description| RViz
    AttnHeatmap --> RViz
    TrajNode -->|/vla/markers @ 50Hz| RViz
    TrajNode -.->|/vla/carried_object_pose| CamNode
```

---

## Dual Pipeline Architecture

The system supports two complementary execution pipelines:

| Feature | Path A: End-to-End VLA (Recommended) | Path B: Modular Grounding + Arc Planner |
| :--- | :--- | :--- |
| **Model** | **Octo-VLA Multimodal Transformer** (`vla_model.py`) | **Language Grounding + DLS Arc Generator** |
| **Visual Input** | Raw $256 \times 256 \times 3$ RGB camera frames | Raw RGB + Camera Intrinsic Ray-Plane Intersection |
| **Language Input** | Word tokens mapped to learned transformer embeddings | Lexical parser with keyword intent matching |
| **Decision Making** | Cross-attention over visual patch tokens | Multi-modal symbol grounding against perceived 3D poses |
| **Action Output** | Continuous 7-DoF delta actions $[\Delta x, \Delta y, \Delta z, \Delta \text{rot}, \text{grip}]$ | Multi-waypoint collision-avoidance 3D trajectory arcs |
| **Explainability** | Real-time cross-attention heatmaps (`/vla/attention_map`) | Explicit Cartesian 3D waypoints & RViz visual ribbon |

---

## Key Capabilities

1. **End-to-End Vision-Language-Action Policy**:
   - Inspired by **Octo** (Open X-Embodiment).
   - Ingests $256 \times 256 \times 3$ RGB images, patches them into 256 tokens, and fuses them with instruction tokens using transformer cross-attention.
   - Evaluates closed-loop actions at $10\,\text{Hz}$ with hardware acceleration (Apple Silicon Metal `mps` on macOS, CPU in Linux VM).

2. **Zero Predefined Object Coordinates**:
   - No static object lists. The overhead camera sensor (`camera_stream_node.py`) renders realistic perspective frames from `camera_link`.
   - The detector node back-projects 2D image contours through camera rays:
     $$\mathbf{r}_{\text{world}} = \mathbf{R}^{\top} \begin{bmatrix} \frac{u - c_x}{f_x} \\ \frac{v - c_y}{f_y} \\ 1 \end{bmatrix}, \quad \mathbf{p}_{\text{world}} = \mathbf{p}_{\text{eye}} + \frac{z_{\text{target}} - z_{\text{eye}}}{r_{z}} \mathbf{r}_{\text{world}}$$
   - Randomizing object locations dynamically via `/vla/randomize_scene` demonstrates true visual adaptation.

3. **Damped Least-Squares (DLS) Numerical IK**:
   - Formulates a $3 \times 6$ numerical Jacobian with Levenberg-Marquardt damping ($\lambda = 0.02$) for singularity-free Cartesian control:
     $$\Delta \mathbf{q} = \mathbf{J}^{\top} \left( \mathbf{J} \mathbf{J}^{\top} + \lambda^2 \mathbf{I}_{3 \times 3} \right)^{-1} \mathbf{e}$$
   - Converges to $< 0.0007\,\text{m}$ Cartesian error in under 50 iterations.

4. **Dynamic TF2 Spatial Reparenting**:
   - **On Table**: Object frame is child of `base_link`.
   - **During Grasp**: Transferred to gripper flange (`tool0`) at $50\,\text{Hz}$.
   - **On Release**: Relocked to `base_link` at the Docking Bay bullseye.
   - Grasp state feeds back to the camera node (`/vla/carried_object_pose`) so the synthetic visual stream reflects the carried payload.

5. **High-Fidelity Studio Digital Twin (RViz2)**:
   - Titanium slate workbench ($1.15\,\text{m} \times 1.35\,\text{m}$), anti-static ESD mat, laser barrier obstacle ($h = 0.20\,\text{m}$), and concentric alloy docking station.
   - Real-time 3D ribbon trajectory visualization and live neural attention heatmaps.

---

## Repository Structure

```
vla-robotic-manipulation/
├── README.md
├── LICENSE
├── test_vla_end_to_end.py        # Standalone VLA verification suite (MPS / CPU)
├── verification/                 # Exported camera views & attention heatmaps
│   ├── camera_view_scene.png
│   └── vla_attention_overlay.png
└── src/
    ├── sesac_robot/              # 6-DOF arm URDF xacro, collision/visual meshes & ros2_control
    ├── vla_bringup/              # Launch orchestration (vla_system.launch.py)
    ├── vla_description/          # Studio workspace RViz configuration (vla_scene.rviz)
    ├── vla_motion_planning/      # DLS IK solver, 50 Hz trajectory executor & VLA action adapter
    │   └── vla_motion_planning/
    │       └── trajectory_executor_node.py
    ├── vla_perception/           # Camera simulation, Octo-VLA model & dynamic perception
    │   └── vla_perception/
    │       ├── camera_stream_node.py   # Synthetic RGB-D camera sensor (15 Hz)
    │       ├── vla_model.py            # Octo-VLA PyTorch Multimodal Transformer
    │       ├── vla_policy_node.py      # Closed-loop 10 Hz VLA inference node
    │       └── detector_node.py        # Dynamic 3D ray back-projection detector
    └── vla_task_planner/         # Natural language task parser & symbol grounding engine
        └── vla_task_planner/
            └── language_task_node.py
```

---

## Verification & Standalone Testing

Run the standalone verification suite without launching the full ROS 2 stack (works natively on macOS Apple Silicon MPS or Linux CPU):
```bash
python3 test_vla_end_to_end.py
```

This verifies:
1. Overhead RGB-D camera rendering ($256 \times 256$ pixels).
2. Octo-VLA PyTorch forward pass on Apple Silicon Metal (MPS) or CPU.
3. Visual cross-attention dynamically localizing target objects from photons alone.
4. Dynamically relocating the red can to randomized unseen coordinates and verifying that the predicted actions update in real time.
5. Saves visual outputs to `verification/camera_view_scene.png` and `verification/vla_attention_overlay.png`.

---

## ROS 2 Quickstart (Ubuntu / UTM VM)

### 1. Build Workspace
```bash
mkdir -p ~/vla_ws/src
cd ~/vla_ws/src
git clone <your-repo-url> .
cd ~/vla_ws
colcon build --symlink-install
source install/setup.bash
```

### 2. Launch the System
```bash
ros2 launch vla_bringup vla_system.launch.py
```
This launches:
- `robot_state_publisher` loading the URDF robot meshes.
- RViz2 with the clean studio workbench, 3D objects, laser barrier, and docking bay.
- `camera_stream_node` streaming real-time RGB and Depth camera frames at $15\,\text{Hz}$.
- `vla_policy_node` running the Octo-VLA PyTorch model in closed loop at $10\,\text{Hz}$.
- `vla_object_detector` dynamically estimating 3D poses from camera photons (zero hardcoding).
- `vla_language_task_planner` awaiting natural language instructions.
- `vla_trajectory_executor` running kinematics and closed-loop VLA action execution at $50\,\text{Hz}$.

---

### 3. Dispatch Natural Language Instructions
Open a second terminal window, source the workspace, and send commands:

#### Test 1: Full Collision-Aware Pick-and-Place (Red Can)
```bash
ros2 topic pub --once /vla/instruction std_msgs/msg/String \
  "data: 'Pick up the red soda can, avoid the laser barrier, and deliver it to Docking Bay 1'"
```
* **Robot Action**: Swings over to the red can, descends, closes gripper, lifts, arcs high ($z = 0.38\,\text{m}$) over the laser barrier, descends onto the docking bay bullseye, releases the can, and returns home.

#### Test 2: Low-Altitude Direct Transfer (Blue Mug)
```bash
ros2 topic pub --once /vla/instruction std_msgs/msg/String \
  "data: 'Take the blue coffee mug and place it on Docking Bay 1'"
```
* **Robot Action**: Transits directly at hover altitude ($z = 0.16\,\text{m}$) without climbing over obstacle height.

#### Test 3: Workspace Reachability Limits (Green Flask)
```bash
ros2 topic pub --once /vla/instruction std_msgs/msg/String \
  "data: 'Transfer the green specimen flask over the laser barrier to the dock'"
```
* **Robot Action**: Reaches the far lateral target ($y = -0.35\,\text{m}$), solves the kinematics cleanly, clears the obstacle, and docks the payload.

#### Test 4: Dynamic Scene Randomization (Zero Predefined Coordinates)
In a separate terminal, trigger dynamic scene randomization:
```bash
ros2 topic pub --once /vla/randomize_scene std_msgs/msg/String "data: 'randomize'"
```
The camera view dynamically updates, the detector re-estimates the 3D centroids from photons, and the VLA policy steers toward the newly placed objects.

---

## License
Distributed under the **MIT License**. See `LICENSE` for details.
