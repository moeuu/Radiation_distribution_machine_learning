docker container run --gpus all --shm-size=8g -itd --rm name docker \
    --net="host" -v $PWD:/project -v /home/morita/src/Radiation_distribution_machine_learning/data -p 8888:8888 docker bash