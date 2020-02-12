close all; clear all; clc


%Calibrazione riferimento con 13 pose ai minimi quadrati
poses_frame_robot = load('poses_frame_robot2');

b = [];

%Prendo il Piano Y-Z
for i=1:length(poses_frame_robot.poses)
    b = [b; poses_frame_robot.poses(i,2)';poses_frame_robot.poses(i,3)'];
end

poses_frame_sheet = load('poses_frame_sheet');
A = [];
for i=1:length(poses_frame_sheet.poses)
    A = [A;
        -poses_frame_sheet.poses(i,1)/100 poses_frame_sheet.poses(i,2)/100 1 0;
        poses_frame_sheet.poses(i,2)/100 poses_frame_sheet.poses(i,1)/100 0 1];
end

x = A\b; %[ cos(theta) sin(theta) Tx Ty]

%Rototanslation Matrix
R_2D = [-x(1) x(2) x(3);
    x(2) x(1) x(4);
    0 0 1]

%Test Point B1 (Sheet Referece)
x_p = [0.0; 0.0; 1.0];
x_r = R_2D * x_p


%Test Point B3
x_p = [0.450; 0.0; 1.0];
x_r = R_2D * x_p






