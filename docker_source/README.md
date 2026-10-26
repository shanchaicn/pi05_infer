# Exported model service source

Copied from the running `pi05-infer-tb6r5_policy_server-1` container
(`openpi_tb6r5_policy_server:orin-jp62`). The Compose service mounts these
directories read-only at their original paths. Dependencies remain in the image.

* `src/openpi/training/config.py`: named policy and data transform configuration.
* `src/openpi/policies/tb6r5_policy.py`: observation and action transforms.
* `src/openpi/policies/policy_config.py`: checkpoint loading and policy creation.
* `scripts/serve_policy.py`: model server entry point.
* `packages/openpi-client/src/openpi_client`: websocket protocol and serialization.

The current `pi05_tb6r5` configuration selects eight action dimensions, and
`TB6R5Outputs` defaults to eight dimensions. The six-axis delta conversion is
still enabled. Check the delta/absolute semantics against the training
configuration before hardware inference with the TCP checkpoint.

`export-sha256.json` records the original image source before local edits.

The `app/` directory outside this export is not the source mounted by Compose.
