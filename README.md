# TB6-R5 pi0.5 inference runtime

For standalone JAX inference in this directory, use `jax_infer.py` and
`compose.jax.yml`; see [the JAX guide](README_JAX.md). This path loads JAX
`params/` explicitly and is independent of the PyTorch host bridge.
The deployment commands below describe the existing PyTorch runtime.

This directory restores the Orin deployment split used by the trained OpenPI
pi0.5 policy:

```text
TB6-R5 state + two cameras
              |
              v
run_pi05_tb6r5.py (host: validation, safety, RPC)
              |
              | ws://127.0.0.1:8002
              v
openpi_tb6r5_policy_server:orin-jp62 (Docker: CUDA inference)
```

The host bridge lives in `openpi_runner.py` in this directory;
`run_pi05_tb6r5.py` remains its command-line entry point. It reuses the robot
SDK and camera helpers from the sibling `tb6r5_policy_infer` package. Set
`TB6R5_PACKAGE_ROOT` if that package is installed elsewhere.

The installed shared package currently provides a LeRobot CLI; its CLI/API
does not provide the OpenPI backend. Use the commands below directly.
Start the model service with `./manage.sh start` before starting the bridge.

The running image's Python source has been exported to `docker_source/` and
mounted read-only at the same paths inside the model container:

| Host path | Container path |
| --- | --- |
| `docker_source/src/openpi` | `/app/src/openpi` |
| `docker_source/scripts` | `/app/scripts` |
| `docker_source/packages/openpi-client/src/openpi_client` | `/app/packages/openpi-client/src/openpi_client` |

Edit these exported files to change the model service. The separate `app/`
tree is not mounted into this container. Restart the model service after Python
source edits; after changing mount definitions, run `./manage.sh start` to
recreate it. `TB6R5Outputs(action_dim=8)` in
`docker_source/src/openpi/training/config.py` selects the first eight action
dimensions. The six-axis delta-action conversion is still enabled and must
be checked against the checkpoint's training configuration before TCP control.

The default checkpoint uses `--action-space ee_pose`:
`[x, y, z, qx, qy, qz, qw, gripper_m]`. The bridge reads the TCP pose from
Topic and builds Cartesian `JogAnyC` commands. A 7D joint checkpoint requires
`--action-space joints` and matching normalization statistics. Dimensions
are checked exactly; outputs are never truncated or padded by the bridge.

`--dry-run` opens only the read-only Topic subscriber and cameras; no robot RPC
connection is opened. HTTP frames are fetched anew for every observation.
Missing/stale feedback, invalid quaternions, out-of-envelope values, and unsafe
position/orientation jumps abort before motion. Black-frame smoke tests only
check the model connection and finite output, and do not validate actions for
hardware use. The bridge does not automatically home the robot.

## Placeholder before training finishes

Do not use a partial `model.safetensors` as a checkpoint placeholder. The
current `/home/ai/pi05_base_pytorch/model.safetensors` is truncated: its file
size is about 1.20 GiB while its safetensors header describes about 13.47 GiB.
`manage.sh check` validates this structural size before starting a real model.

For API, Docker, websocket, and action-validation testing, start the safe mock
policy instead:

```bash
cd /home/ai/pi0.5/pi05_infer
./manage.sh placeholder-start
curl --noproxy '*' http://127.0.0.1:8002/healthz
```

The placeholder returns a ten-step action chunk that holds the current robot
state. Its websocket metadata contains `placeholder=true`, and the host
runner refuses it unless `--dry-run` is active. It cannot be enabled for real
hardware even when the normal hardware confirmation string is supplied.

Stop it before switching to a completed checkpoint:

```bash
./manage.sh placeholder-stop

# The active inference checkpoint is mounted from /home/ai/pi0.5/model/pytorch/pp_white_red/19999/:
#   model.safetensors
#   assets/norm_stats.json

./manage.sh check
./manage.sh start
./manage.sh status
```

After the real service is healthy, run a no-camera dry-run first and then a
real-camera dry-run before considering hardware control.

The Docker image already contains OpenPI and the Jetson CUDA dependencies. It
does not contain a checkpoint. The active checkpoint directory is
`/home/ai/pi0.5/model/pytorch/pp_white_red/19999/` and must contain:

```text
/home/ai/pi0.5/model/pytorch/pp_white_red/19999/model.safetensors
/home/ai/pi0.5/model/pytorch/pp_white_red/19999/assets/norm_stats.json
```

The host safety validator uses the same directory by default. To temporarily
use another checkpoint's normalization statistics, pass `--norm-stats` to
`run_pi05_tb6r5.py` or set `TB6R5_OPENPI_POLICY_DIR` for that command.

`metadata.pt` and `optimizer.pt` are training files and are not required for inference.

## 1. Check and start the policy server

With the new checkpoint in `/home/ai/pi0.5/model/pytorch/pp_white_red/19999/`:

```bash
cd /home/ai/pi0.5/pi05_infer
./manage.sh check
./manage.sh start
./manage.sh status
```

To use a different checkpoint for one command without editing configuration,
set `CHECKPOINT_HOST_DIR`. The container mount and `manage.sh` validation use
the same value:

```bash
cd /home/ai/pi0.5/pi05_infer
CHECKPOINT_HOST_DIR=/absolute/path/to/checkpoint ./manage.sh restart
```

The Compose definition uses the existing image and never pulls or rebuilds it.
Model loading can take several minutes; `manage.sh start` waits for `/healthz`.

Useful operations:

```bash
./manage.sh logs
./manage.sh restart
./manage.sh stop
```

## 2. Safe pipeline tests

Use the Python 3.12 environment because its ABI matches the bundled TB6-R5
RPC/Topic extensions. The old `/home/ai/condaenv/openpi` environment is Python
3.11 and must not be used for real hardware.

Local dependencies and SDK only:

```bash
/home/ai/condaenv/openpi_orin/bin/python3 run_pi05_tb6r5.py \
  --local-check --no-camera
```

Connect to the model and validate one response without opening a robot RPC
connection:

```bash
/home/ai/condaenv/openpi_orin/bin/python3 run_pi05_tb6r5.py \
  --dry-run \
  --smoke-test \
  --no-camera \
  --no-show-camera \
  --first-action-timeout-s 120
```

Black-frame output is meaningless; this command only verifies the complete
Docker/GPU/websocket response path. It accepts only a finite, non-empty action
array and does not apply TB6-R5 action limits or process any returned action.
`--smoke-test` automatically enables `--dry-run` and requires `--no-camera`
without `--print-rpc`. Omitting `--dry-run` does not enable robot control.
It runs once by default; set `--max-inference-calls` to validate more responses.

To diagnose the action-dimension contract, run the dedicated live test after
starting the policy service. It sends black frames only and never opens a robot
RPC connection:

```bash
cd /home/ai/pi0.5/pi05_infer
PI05_RUN_LIVE_POLICY_DIMENSION_TEST=1 \
  /home/ai/condaenv/openpi_orin/bin/python3 -m unittest \
  discover -s tests -p 'test_runner.py' -v
```

The test prints the raw action dimension returned by the service alongside the
dimension declared in the mounted checkpoint's `assets/norm_stats.json`.

Then validate the two default HTTP cameras, still without robot control:

```bash
/home/ai/condaenv/openpi_orin/bin/python3 run_pi05_tb6r5.py \
  --dry-run \
  --print-rpc \
  --no-show-camera \
  --max-inference-calls 2 \
  --task "Pick up the yellow ring and place it on the white plate."
```

By default the cameras are read from the robot at:

```text
realsense_0=http://ROBOT_IP:8888/RsCameraSensor/1/0/color
realsense_1=http://ROBOT_IP:8888/RsCameraSensor/0/0/color
```

Use `--camera-devices` for V4L2 or `--camera-serials` for local RealSense.
Local RealSense serial mode additionally needs a Jetson-compatible
`pyrealsense2`; HTTP and V4L2 modes do not.

During every OpenPI inference, the runner prints the Topic `robottarget` TCP
pose, `gripper_mm`, and `gripper_joint_m` used by the policy, followed by the
complete raw policy action chunk before safety validation. Add `--print-rpc`
to additionally print the exact outgoing controller payload.

## 3. Real hardware gate

Real hardware is disabled unless the explicit confirmation is present. Only
remove `--dry-run` after checking the model outputs, camera mapping, task text,
home pose, controller dialect, and gripper model:

```bash
/home/ai/condaenv/openpi_orin/bin/python3 run_pi05_tb6r5.py \
  --confirm-hardware TB6R5_REAL_INFERENCE \
  --robot-ip 192.168.2.174 \
  --task "Pick up the yellow ring and place it on the white plate." \
  --no-show-camera
```

Current host bridge defaults:

- `g_model=3`
- `cd_version=45`
- `subloop=0`
- 30 Hz control, four actions executed from each returned chunk
- 0.03 rad maximum commanded joint change per step
- 0.03 m maximum TCP position change and 0.15 rad maximum rotation per step
- 30–70 mm gripper range

Every observation and action chunk is checked for shape, finite values,
training-envelope limits, and unsafe jumps. Runtime failures trigger SubLoop
exit and robot Disable before the process exits.
