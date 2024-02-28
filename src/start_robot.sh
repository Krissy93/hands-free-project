#!/bin/bash

path=$(rosparam get hands_free_node/path)
cd "$path"
cd ..

./intera.sh sim | roslaunch sawyer_gazebo sawyer_world.launch
