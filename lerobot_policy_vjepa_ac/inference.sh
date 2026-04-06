python -m lerobot.async_inference.robot_client \
--lerobot.type=so_follower \
--robot.port=/dev/ttyACM0 \
--robot.cameras='{"top": {"type": "opencv", "index_or_path": 0, "width": 640", "height": 480, "fps": 30}}' \
--robot.id=orange \
--task="Place the scotch in the box" \
--policy.type=vjepa_ac \
--pretrained_name_or_path=outputs/vjepa_ac/run_20260406_overfit_4/checkpoints/last \
--policy_device=cuda \
--actions_per_chunk=15