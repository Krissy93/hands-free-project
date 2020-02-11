import numpy as np
import cv2
import rospy
from intera_core_msgs.msg import EndpointState

global point


class color:
   PURPLE = '\033[95m' #loginfo and commands
   CYAN = '\033[96m' #quando chiedo input
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m' #ok
   YELLOW = '\033[93m' #info
   RED = '\033[91m' #errori
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def callback(msg, file):
    #point = msg
    print(color.BOLD + color.GREEN + 'Acquired point is: ' + color.END)
    print('(' + str(msg.pose.position.x) + ', ' + str(msg.pose.position.y) + ', ' + str(msg.pose.position.z) + ')')
    file.write(str(msg.pose.position.x) + ', ' + str(msg.pose.position.y) + ', ' + str(msg.pose.position.z) + '\n')


def myhook():
    rospy.loginfo(color.BOLD + color.RED + '\n -- KEYBOARD INTERRUPT, SHUTTING DOWN --' + color.END)

def main():
    ''' Pubblica le coordinate dell'indice in tempo reale in un topic. Quando necessario
    manda le coordinate di movimento al robot. Pubblicare a video i due workspace come
    acquisizione RGB in tempo reale '''

    rospy.init_node('robot_workspace_calibration_node')
    posenode = '/robot/limb/right/endpoint_state'

    raw_input(color.BOLD + color.YELLOW + '-- Starting! Press any key to continue' + color.END)
    with open("robot_vertical_points.txt", "a+") as file:
        while not rospy.is_shutdown():
            #sub = rospy.Subscriber(posenode, EndpointState, callback, queue_size=1)
            msg = rospy.wait_for_message(posenode, EndpointState)
            callback(msg, file)
            raw_input(color.BOLD + color.YELLOW + 'Press any key to continue' + color.END)


        file.close()
    rospy.on_shutdown(myhook)

if __name__ == '__main__':
    main()


'''A1 = 0.7918739155, 0.299388204446, 0.655044483057
   A2 = 0.800361982553, 0.0745365382896, 0.652130100878
   A3 = 0.808782030742, -0.141144171443, 0.649997919078
   A4 = 0.797298454516, 0.189626348211, 0.479352486995
   A5 = 0.804667474232, -0.0364615281983, 0.476384063774
   O1 = 0.794093654732, 0.304803141793, 0.306267654831
   O0 = 0.80279146887, 0.0795482317203, 0.303469836724
   O2 = 0.813292720894, -0.146383434076, 0.300444352477
   B4 = 0.800810621971, 0.194802139985, 0.129312335931
   B5 = 0.80972621126, -0.0309495529717, 0.127465293553
   B1 = 0.797267155506, 0.311215362935, -0.0436766055149
   B2 = 0.805927033876, 0.0838474027253, -0.0453659637392
   B3 = 0.816420878966, -0.14130168333, -0.0492138422795
   '''
