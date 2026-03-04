echo Free the RAM and restart Cosmos Reason2 2B model
docker stop cosmos-reason2-2b
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
sleep 2
docker start cosmos-reason2-2b
