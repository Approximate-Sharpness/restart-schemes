% radial sampling scheme in 2 dimensions

function mask = sa_radial_2d(N, lines)

mask = zeros(N,N,'logical');

for l=1:lines
    theta = 2*pi*(l-1)/lines;
    xdir = N*cos(theta);
    ydir = N*sin(theta);

    if (xdir >= 0) && (ydir >= 0)
        len = sqrt(min(xdir,N/2)^2 + min(ydir,N/2)^2);
    elseif (xdir >= 0) && (ydir < 0)
        len = sqrt(min(xdir,N/2)^2 + max(ydir,-N/2+1)^2);
    elseif (xdir < 0) && (ydir >= 0)
        len = sqrt(max(xdir,-N/2+1)^2 + min(ydir,N/2)^2);
    else % dx < 0 && dy < 0
        len = sqrt(max(xdir,-N/2+1)^2 + max(ydir,-N/2+1)^2);
    end
    len = N;
    mask = draw_line(mask,N/2,N/2,N/2+len*xdir/N,N/2+len*ydir/N);

end

end


function arr = draw_line(arr,x1,y1,x2,y2)
% draw a line using the digital differential analyzer algorithm

dx = x2-x1;
dy = y2-y1;

if (abs(dx) >= abs(dy))
    step = abs(dx);
    dxflag = true;
else
    step = abs(dy);
    dxflag = false;
end

if dxflag
    dx = round(dx/step);
    dy = dy/step;
else
    dx = dx/step;
    dy = round(dy/step);
end

x = round(x1);
y = round(y1);
i = 1;
while (i <= step)
    if dxflag
        xrd = x;
        yrd = round(y);
    else
        xrd = round(x);
        yrd = y;
    end
    
    if (xrd < 1) || (xrd > size(arr,1)) || (yrd < 1) || (yrd > size(arr,2))
        break;
    else
        arr(xrd,yrd) = 1;
    end
        
    x = x + dx;
    y = y + dy;
    i = i + 1;
end

end
