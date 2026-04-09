fileID = fopen('../Outputs/my_photon_output.dat','r');
formatSpec = '%e %e %e %e %e %e %e %e %e %e %e\n';
sizeA = [11 100];
data1 = fscanf(fileID,formatSpec,sizeA); %columns represent diff phase angles
fclose(fileID);
phase1 = data1(1, :);
I1 = data1(2, :);
Q1 = data1(3, :);
U1 = data1(4, :);
V1 = data1(5, :);

% Mini-RF uses [1 0 0 -1] as RCP, literature + Parvathy use [1 0 0 1],
% which causes SC to be I+V and OC to be I-V
% can double check by converting electric field in MC code to stokes parameters 
SC1 = (I1+V1);
OC1 = (I1-V1);
CPR1 = SC1./OC1;

x = [0.1:0.1:10];
figure;plot(x, CPR1, 'LineWidth', 5);
hold on; ylabel('CPR', 'FontSize', 25);
hold on; xlabel('Phase', 'FontSize', 25);
ax = gca;
exportgraphics(ax,'my_plot.jpg','Resolution',600)