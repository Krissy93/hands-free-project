#!/bin/bash

# Ottieni il percorso specificato nel parametro ROS
path=$(rosparam get hands_free_node/path_ur)

# Cambia la directory corrente al percorso ottenuto
cd "$path"
cd ..

# Funzione per avviare roslaunch su un nuovo terminale in modo supportato
function launch_in_new_terminal {
    local cmd="$1"
    local terminal_cmd="roslaunch $cmd"
    gnome-terminal -- bash -c "$terminal_cmd; exec bash"
    sleep 10  # Aggiungi un piccolo ritardo prima di avviare il prossimo roslaunch
}

# Avvia i launch dei comandi per UR3 nell'ordine richiesto su terminali separati
launch_in_new_terminal "roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=192.168.1.100"
launch_in_new_terminal "ur3_moveit_config moveit_planning_execution.launch"
launch_in_new_terminal "ur3_moveit_config moveit_rviz.launch"

# Attendere che tutti i processi dei launch abbiano completato l'avvio
#wait
