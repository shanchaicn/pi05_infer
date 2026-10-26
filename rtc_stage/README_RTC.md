# π0.5 RTC staging on Jetson

This staged implementation adds guided Real-Time Chunking to the π0.5 policy
on both backends. PyTorch sampling lives in `openpi/models_pytorch/pi0_pytorch.py`
(`sample_actions_rtc`). JAX sampling lives in `openpi/models/pi0.py` and is
served by `jax_infer.py` (default port 8003), which loads this `rtc_stage`
tree. The original port 8002 service is unchanged. The staged PyTorch service
uses port 8012 and the same `pi05_pp_white_red_fft` checkpoint.

`Policy.metadata["rtc_supported"]` is true for either backend when the loaded
model implements `sample_actions_rtc`. The host runners (`openpi_runner.py` and
`rtc_stage/openpi_runner.py`) accept that flag from a JAX or a PyTorch server.
`--offline-dry-run` exercises recording and RTC without a robot.

The policy accepts a WebSocket observation with an optional `rtc` object:

```python
{"rtc": {
    "previous_actions": previous_chunk[executed_steps:],  # physical 8D actions
    "executed_steps": executed_steps,
    "delay_steps": predicted_inference_delay_in_control_ticks,
    "num_steps": 5,
    "guidance_beta": 5.0,
}}
```

The server normalizes the previous actions with the checkpoint statistics,
pads them to the model's 32D action width, and applies soft-mask VJP guidance
at each flow step. The host runner executes the active chunk while the next
one is inferred, then skips the new chunk actions whose control ticks already
passed. It aborts if inference outlasts the active chunk or any existing
state/action safety check fails. It does not suppress errors or repeat the
last action indefinitely.

## Validation completed

- Unit tests: `python -m pytest -q test_rtc.py test_rtc_runner.py` (4 passed).
- Model-only WebSocket check: initial and guided responses were both 10×8;
  after warmup, the initial call took 0.716 s and the guided call 1.224 s.
- In that check, the four frozen TCP positions differed from the previous
  chunk by 0.8–2.2 mm. The initial cold call took 14.5 s, so the runner
  warms the policy before opening the robot command channel.

The staged host runner requires the explicit
`--confirm-hardware TB6R5_REAL_INFERENCE` argument before sending RPC.
For real hardware, use `/home/ai/condaenv/tb6r5/bin/python` (Python 3.10);
the ARM RPC SDK does not connect reliably from `openpi_orin` (Python 3.12).

At `--fps 2`, the 10-step horizon lasts 5 seconds. This is a mechanism
validation setting and changes the time scale from the prior 30 Hz runner.
It does not prove task success or fix a gripper prediction that stays open.

The staged host runner can be invoked with the verified front/wrist mapping:

```bash
cd /home/ai/pi0.5/pi05_infer/rtc_stage
/home/ai/condaenv/tb6r5/bin/python run_pi05_tb6r5.py \
  --rtc --rtc-num-steps 5 --fps 1 --record-fps 1 \
  --confirm-hardware TB6R5_REAL_INFERENCE \
  --robot-ip 192.168.2.151 --policy-port 8012 \
  --camera-urls 'realsense_0=http://192.168.2.151:6888/RsCameraSensor/0/0/color,realsense_1=http://192.168.2.151:6888/RsCameraSensor/1/0/color' \
  --no-show-camera \
  --task 'Pick up the white object and place it on the red object.' \
  --max-inference-calls 2 --print-rpc \
  --record-dataset /home/ai/pi0.5/data/lerobotdata/pi05_rtc_new_run
```

The recording directory must be new and empty. The LeRobot v3 writer saves
front/wrist images, 8D pre-command state and the 8D action after its RPC send
succeeds. `--record-episode-frames` defaults to 100, and shutdown saves the
remaining frames. The data includes only control steps that were sent, not
model actions skipped during RTC overlap. Do not run another motion runner
against port 5868 at the same time.

On 2026-10-26, a bounded 1 Hz hardware test completed two inference chunks
and 12 control steps. The guided chunk took 1.979 s and skipped two overlap
actions. The resulting v3 dataset contained one 12-frame episode at
`/home/ai/pi0.5/data/lerobotdata/pi05_rtc_hardware_20261026_2` and was read
back successfully. `--max-inference-calls 0` permits an unlimited run after
a bounded observation. All existing state-envelope and per-action movement
checks remain active.
