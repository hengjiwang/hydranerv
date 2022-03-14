%Practice how to make Raster Plot

%t1= spike(1).times; % access the spike times for the first and second trials
%t2= spike(2).times;


figure          %create a new figure
hold on         % multiple plots on the same figure
for i = 1:length(cb_locs_time)            %loop through each spiketime
    line([cb_locs_time(i) cb_locs_time(i)], [1 2])  % tick mark at x = t1(i) with a height of 1
end

ylim([0 2])   % reformat y-axis for legibility, length of the y axis
xlim([0 t(end)]) %range of x axis
xlabel('Time(min)')
%ylabel('Trial Number')
set(gca,'YTick',[])   % remove ytick

for i = 1:length(rp_locs_time)
    line([rp_locs_time(i) rp_locs_time(i)],[0 1],'Color','red')
end
