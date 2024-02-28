COME USARE QUESTO SOFTWARE

- installare UBUNTU FOCAL 20
- installare ROS1 NOETIC http://wiki.ros.org/Installation/Ubuntu
- installare CUDA e cuDNN https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux
- installare OPENCV https://linuxize.com/post/how-to-install-opencv-on-ubuntu-20-04/
- installare MediaPipe https://google.github.io/mediapipe/getting_started/install.html#installing-on-debian-and-ubuntu

----

PRIMO STEP: CALIBRARE IL WORKSPACE UTENTE

- calibrate_camera.py
  E' settata per funzionare sia con Kinect che con una camera qualsiasi che si apre con VideoCapture. Contiene due sottoprogrammi:
    1) routine per acquisire le immagini del workspace da usare per la calibrazione della camera.
    2) routine da usare subito dopo l'acquisizione per calibrare la camera ed estrarre:
    la matrice intrinseca della camera, i valori di distorsione delle lenti, R e t relative al sistema di riferimento centrato nell'origine della chessboard
    Queste servono per convertire dai pixel (nel frame camera) ai metri (nel frame utente) centrati nell'origine del sistema di riferimento.
  A seconda della modalita' di lancio puo' essere usata solo per acquisire, solo per calibrare o la routine completa.

---> come si fa?
1) stampo un master di calibrazione camera (chessboard). La grandezza dei quadretti dipende dalla risoluzione della camera: piu' e' spinta, piu' possono essere piccoli
2) incollo il master su un supporto rigido
3) lancio calibrate_camera.py in modalita' ACQUISIZIONE o COMPLETA. Usandolo, acquisisco almeno 30 immagini della chessboard in diverse posizioni, anche inclinata e non solo parallela al piano
4) (solo se non ho lanciato la completa prima) lancio calibrate_camera.py in modalita' CALIBRAZIONE dando come input la cartella contenente le immagini appena acquisite

SECONDO STEP: CALIBRARE IL ROBOT

- calibrate_robot.py
  Programma che guida l'utente nell'acquisizione dei punti robot corrispondenti ai marker del master sul quale opera il robot. L'utente deve prima definire i punti del master e le loro coordinate nel sistema di riferimento definito da lui (centrato in una origine 0,0).
  Usando il centering tool o altri sistemi per posizionare correttamente il robot, si posiziona il suo end effector in corrispondenza del marker e si acquisisce la posa cartesiana dal suo ROS node.
  Il risultato e' un file yaml che contiene i marker dell'utente, i corrispondenti nel sistema robot e infine la matrice di calibrazione R

---> come si fa?
1) stampo un master di calibrazione robot. Questo puo' essere qualsiasi, ad esempio con i marker tondi posizionati a intervalli definiti dall'utente.
2) lo incollo su un supporto rigido e lo fisso, in modo che rimanga sempre lo stesso. Questo corrisponde al workspace sul quale operera' il robot.
3) annoto in un file yaml (ad esempio master_workspace.yaml) le coordinate dei marker secondo il sistema di riferimento definito dall'utente
4) lancio calibrate_robot.py: devo posizionare il robot manualmente nei punti definiti in master_workspace infilando l'end-effector all'interno del centering tool in modo che rimanga a contatto con la superficie.
   NOTA: questa procedura implica contatto tra le superfici, di conseguenza il workspace potrebbe variare dunque si introduce un certo errore di calibrazione dovuto alla procedura manuale. Sarebbe meglio averlo completamente visivo (TODO)

FUNZIONI ACCESSORIE

- cartesian.py
  contiene la funzione move2cartesian del Sawyer rivisitata in modo da poter essere chiamata da codice

- conversion_utils.py
  libreria che contiene le funzioni di conversione tra un sistema di riferimento all'altro. Queste sono:
  px2meters -> converte da pixels a metri rispetto al workspace calibrato della camera
  H2R -> converte dal workspace utente H al workspace intermedio W e infine da W al sistema robot R, in modo da passare in una sola botta i punti H al robot R
  px2R -> funzione che combina le precedenti, convertendo una lista di punti in px in una equivalente lista di punti nel sistema R. Se la lista contiene un solo punto va bene lo stesso!

- graphical_utils.py
  libreria che contiene funzioni di tipo grafico, tra cui Colors per plottare a terminale i log colorati e una serie di funzioni di visualizzazione, tipo quella che mostra lo scheletro della mano a video
