clear
%% set file parameters
movieParam.filePath = '\'; % path to your file, with a slash in the end
movieParam.fileName = ''; % file name, must be .tif format
minfo = imfinfo([movieParam.filePath movieParam.fileName '.tif']); % this line reads in file information
movieParam.numImages = length(minfo); 
movieParam.imageSize = [minfo(1).Height minfo(1).Width];

ifvisualize = 1; % do you want to visualize the result and save it as a video? if so, it will be saved as out.avi under the current folder

framerate = 2; %fps



%% segmentation
[bw,a,b,cent,theta] = fitGcampEllipse(movieParam,ifvisualize);

save('ellipse_workspace.mat');

%create time vector
endtime = (1/framerate) * (movieParam.numImages-1) ./ 60;
t = linspace(0,endtime,movieParam.numImages);



% plot result
figure;
plot(t,a,'r'); hold on; plot(t,b,'k')
ylabel('pixels')
legend('length','width')

