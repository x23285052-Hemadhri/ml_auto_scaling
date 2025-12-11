# Loop CPU load: 60s stress, 60s cooldown
while true; do
  echo "[INFO] CPU Load for 60s"
  stress-ng --cpu 2 --timeout 60

  echo "[INFO] Memory Load for 60s"
  stress-ng --vm 1 --vm-bytes 500M --timeout 60

  echo "[INFO] I/O Load for 60s"
  stress-ng --io 2 --timeout 60

  echo "[INFO] Cooling down for 60s"
  sleep 60
done

echo "Workload simulation complete!"