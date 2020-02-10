# loads from yaml file
# first file: robot points that calibration saves
# second file: reference master markers positions

import yaml
import numpy as np

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

def yaml2dict(path):
    with open(path, 'r') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        rospy.loginfo(color.BOLD + color.YELLOW + 'Reading YAML file...' + color.END)

        dictionary = yaml.load(file, Loader=yaml.FullLoader)

        return dictionary

def dict2yaml(dictionary, path):

#dict_file = [{'sports' : ['soccer', 'football', 'basketball', 'cricket', 'hockey', 'table tennis']},
#{'countries' : ['Pakistan', 'USA', 'India', 'China', 'Germany', 'France', 'Spain']}]

#nota: se ho una lista di dizionari mi mette il file per categorie

    with open(path, 'w') as file:
        result = yaml.dump(dictionary, file)

        rospy.loginfo(color.BOLD + color.GREEN + 'YAML file saved!' + color.END)

def loadcalibcamera(path):
    '''Function that loads the calibration yaml file and returns calibration matrixes as arrays.
    Check that the calibration from calibrate.py has been saved as:
    dict = [{'K' : [[],[],[]]}, {'D' : [[]]}, {'R' : [[],[],[]]}, {'t' : [[]]}, {'Rd' : [[],[],[]]}, {'td' : [[]]}]'''

    dictionary = yaml2dict(path)

    K = dictionary[0]['K']
    K = np.asarray(K)
    D = dictionary[0]['D']
    D = np.asarray(D)
    R = dictionary[0]['R']
    R = np.asarray(R)
    t = dictionary[0]['t']
    t = np.asarray(t)
    Rd = dictionary[0]['Rd']
    Rd = np.asarray(Rd)
    td = dictionary[0]['td']
    td = np.asarray(td)

    return K, D, R, t, Rd, td

def loadcalibrobot(path):
    '''Function that loads the calibration yaml file and returns calibration matrixes as arrays.
    Check that the calibration from vertical_calibration.py has been saved as:
    dict = [{'Master' : [[x, y], [x, y]]}, {'Robot' : [[x, y], [x, y]]}]'''

    dictionary = yaml2dict(path)

    Master = dictionary[0]['Master']
    #Master = np.asarray(Master)
    Robot = dictionary[0]['Robot']
    #Robot = np.asarray(Robot)

    # ho caricato le due matrici come array, ora devo tirare fuori la calibrata
    # Master va trasformata in questa:
    # Mx -My 1 0; My Mx 0 1 per ogni riga di Master originale
    A = []
    for i in range(0, len(Master)):
        # righe di Master composte da coppie x y
        row1 = [Master[i][0], -Master[i][1], 1, 0]
        row2 = [Master[i][1], Master[i][0], 0, 1]
        A.append(row1)
        A.append(row2)
    A = np.asarray(A)

    # vettore verticale fatto cosi' [x1 y1 x2 y2 ecc]
    # attenzione che siccome si tratta del piano ZY devo mettere prima z poi y
    b = []
    for i in range(0, len(Robot)):
        b.append(Robot[i][2])
        b.append(Robot[i][1])
    b = np.asarray(b)

    # solve linear system
     x = np.linalg.lstsq(A,b,rcond=None)
     x = x[0].tolist()
     # shape should be 4,1

     # define rototranslation matrix for robot
     R = [[x[0][0], x[0][1], x[0][2]], [x[0][1], x[0][0], x[0][3]], [0.0,0.0,1.0]]
     R = np.asarray(R)

     return R

'''
per la calibrazione videocamera devo salvare lo yaml cosi'
dictionary = [{'K': mtx.tolist()}, {'D' : dist.tolist()}, {'R' : R1.tolist()}, {'t' : tvecs1.tolist()}, {'Rd' : R2.tolist()}, {'td' : tvecs2.tolist()}]
dict2yaml(dictionary, 'cameracalibration.yaml')

RICORDATI DI METTERE RETURN R DA FINDRT

per la calibrazione robot devo editare vertical_workspace_calibration.py
togli il file write e metti un list append
L = [] fuori dal while, togliere il with open
nella callback
L.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
check che L sia [[],[],[],[]] ecc

fuori dal while not rospy shutdown
robot = {'Robot' : L}
master = {'Master' : [[0.0, 70], [22.5, 70.0], [45.0, 70.0], [11.25, 52.5], [33.75, 52.5], [0.0, 35.0], [22.5, 35.0], [45.0, 35.0], [0.0, 0.0], [22.5, 0.0], [45.0, 0.0], [11.25, 17.5], [33.75, 17.5]]}
dictionary = [master, robot]
dict2yaml(dictionary, 'robotworkspacecalibration.yaml')
'''
