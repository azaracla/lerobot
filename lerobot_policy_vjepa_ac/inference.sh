python -m lerobot.async_inference.robot_client \
--lerobot.type=so_follower \
--robot.port=/dev/ttyACM0 \
--robot.cameras='{"top": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
--robot.id=orange \
--task="Place the scotch in the box" \
--policy.type=vjepa_ac \
--pretrained_name_or_path=outputs/vjepa_ac/run_20260406_overfit_4/checkpoints/last \
--policy_device=cuda \
--actions_per_chunk=15


lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_robot \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --display_data=false \
  --dataset.repo_id=${HF_USER}/eval_act_so101_pickup \
  --dataset.num_episodes=10 \
  --dataset.single_task="Put the scotch in the transparent box" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.vcodec=auto \
  --policy.path=${HF_USER}/act_policy

python -m lerobot.async_inference.robot_client \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM0 \
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
--robot.id=orange \
--task="Place the scotch in the box" \
--policy_type=act \
--pretrained_name_or_path=${HF_USER}/act_policy  \
--policy_device=cuda \
--server_address=192.168.1.128:9001 \
--actions_per_chunk=15 