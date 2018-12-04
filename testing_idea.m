%% Videoreader
v = VideoReader('resources/corridor_5_fps.mp4');
npeaks = 6;
while hasFrame(v)
    video = readFrame(v);
    subplot(2,2,[1,3])
    I  = rgb2gray(video);
    BW = edge(I,'canny');
    [H,T,R] = hough(BW);
    imshow(H,[],'XData',T,'YData',R,...
                'InitialMagnification','fit');
    xlabel('\theta'), ylabel('\rho');
    axis on, axis normal, hold on;

    P  = houghpeaks(H,npeaks,'threshold',ceil(0.3*max(H(:))));
    x = T(P(:,2)); y = R(P(:,1));
    plot(x,y,'s','color','white');
    axis on, axis normal
    subplot(2,2,2)
    lines = houghlines(BW,T,R,P,'FillGap',5,'MinLength',7);
    imshow(I), hold on
    max_len = 0;
    for k = 1:length(lines)
       xy = [lines(k).point1; lines(k).point2];
       plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

       % Plot beginnings and ends of lines
       plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
       plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

       % Determine the endpoints of the longest line segment
       len = norm(lines(k).point1 - lines(k).point2);
       if ( len > max_len)
          max_len = len;
          xy_long = xy;
       end
    end
    pause(0.3);
end