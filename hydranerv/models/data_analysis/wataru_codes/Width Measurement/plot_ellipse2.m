function [x,y] = plot_ellipse2(s,ifplot)
% s - struct returned by regionprops

if nargin<2
    ifplot = true;
end

phi = linspace(0,2*pi,360);
cosphi = cos(phi);
sinphi = sin(phi);

xbar = s.Centroid(1);
ybar = s.Centroid(2);

a = s.MajorAxisLength/2;
b = s.MinorAxisLength/2;

theta = pi*s.Orientation/180;
R = [ cos(theta)   sin(theta)
     -sin(theta)   cos(theta)];

xy = [a*cosphi; b*sinphi];
xy = R*xy;

x = xy(1,:) + xbar;
y = xy(2,:) + ybar;

if ifplot
    plot(x,y,'r','LineWidth',2);
end

end